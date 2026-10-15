// Amber compute conformance runner: executes the compute subset of Khronos VK-GL-CTS Amber scripts with spirv2air
// on Metal and classifies every script as PASS, FAIL, TRANSLATE_REJECT, UNSUPPORTED_AMBER, CRASH or HANG.
//
// The CTS data is external (Apache-2.0) and is never part of this repository; pass the directory that holds the
// .amber files (external/vulkancts/data/vulkan/amber in a VK-GL-CTS checkout) with --cts.
//
// Supported Amber subset: SHADER compute (GLSL via glslangValidator, SPIRV-ASM via spirv-as), BUFFER with DATA_TYPE
// (scalars, vecN, matCxR), SIZE + FILL / SERIES_FROM, DATA ... END, PIPELINE compute with ATTACH and BIND BUFFER AS
// storage / uniform / push_constant, RUN pipeline x y z, EXPECT buffer IDX byte [SIZE n] [TOLERANCE t...] EQ values,
// EXPECT a EQ_BUFFER b. DEVICE_FEATURE / DEVICE_EXTENSION / DEVICE_PROPERTY are parsed and ignored. Anything else
// (images, samplers, graphics, CLEAR, COPY, ...) is reported as UNSUPPORTED_AMBER with the feature name; the shaders
// are still compiled and translated so the detail also shows whether spirv2air accepts them.
//
// Each script runs in a child process (this binary with --one) under a timeout; the driver mode classifies a
// signalled child as CRASH and a timed-out child as HANG.
//
// usage: amber_runner --cts DIR [--all-compute | --list FILE | FILE.amber...] [--csv OUT.csv] [--timeout SEC]
//                     [--offset N] [--limit N] [--tmp DIR] [--inline]
// --inline runs spirv-opt (inline calls, drop dead functions) between the front end and spirv2air.
//        amber_runner --cts DIR --one FILE.amber        (single script, prints one RESULT line)
// Metal buffer indices are assigned in BIND order; the push constant block gets the next index after the buffers.
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <mach-o/dyld.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

extern char** environ;

namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------------------------------------------

enum class Status { Pass, Fail, TranslateReject, UnsupportedAmber, Crash, Hang };

const char* statusName(Status s) {
  switch (s) {
    case Status::Pass: return "PASS";
    case Status::Fail: return "FAIL";
    case Status::TranslateReject: return "TRANSLATE_REJECT";
    case Status::UnsupportedAmber: return "UNSUPPORTED_AMBER";
    case Status::Crash: return "CRASH";
    case Status::Hang: return "HANG";
  }
  return "?";
}

struct Stop {  // thrown to end a script with a classification
  Status status;
  std::string detail;
};

std::string oneLine(std::string s, size_t limit = 600) {
  for (char& c : s)
    if (c == '\n' || c == '\r' || c == '\t') c = ' ';
  if (s.size() > limit) s = s.substr(0, limit) + "...";
  return s;
}

std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string format(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  return buf;
}

// ---------------------------------------------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------------------------------------------

std::vector<std::string> splitWords(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream in(s);
  for (std::string w; in >> w;) out.push_back(w);
  return out;
}

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
  return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

std::vector<std::string> splitLines(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream in(s);
  for (std::string l; std::getline(in, l);) out.push_back(l);
  return out;
}

std::string stripComment(const std::string& line) {
  size_t p = line.find('#');
  return p == std::string::npos ? line : line.substr(0, p);
}

std::string readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// ---------------------------------------------------------------------------------------------------------------
// Element types and value encoding
// ---------------------------------------------------------------------------------------------------------------

enum class Kind { Uint, Int, Float };

struct DataType {
  Kind kind = Kind::Uint;
  unsigned bits = 32;
  unsigned comps = 1;  // scalars per element: vecN -> N, matCxR -> C*R
  unsigned rows = 1;   // vector size: vecN -> N, matCxR -> R (each of the C columns is one vector)
  unsigned scalarBytes() const { return bits / 8; }
  // std430, the default layout: a 3-component vector (or matrix column) occupies the slots of 4.
  bool padded() const { return rows == 3; }
  unsigned elementBytes() const { return (padded() ? comps / 3 * 4 : comps) * bits / 8; }
  // Whether scalar slot i of the padded layout is padding.
  bool isPad(size_t i) const { return padded() && i % 4 == 3; }
};

// Spreads tightly packed scalars into the std430 layout of t (a zero after every third scalar of a 3-vector).
void padLayout(const DataType& t, std::vector<uint8_t>& bytes) {
  if (!t.padded()) return;
  unsigned sb = t.scalarBytes();
  std::vector<uint8_t> out;
  out.reserve(bytes.size() / 3 * 4 + sb);
  for (size_t i = 0; i < bytes.size(); i += sb) {
    out.insert(out.end(), bytes.begin() + i, bytes.begin() + std::min(bytes.size(), i + sb));
    if (i / sb % 3 == 2) out.resize(out.size() + sb, 0);
  }
  bytes.swap(out);
}

bool parseScalarType(const std::string& s, DataType& t) {
  struct Entry { const char* name; Kind kind; unsigned bits; };
  static const Entry table[] = {{"uint8", Kind::Uint, 8},   {"uint16", Kind::Uint, 16}, {"uint32", Kind::Uint, 32},
                                {"uint64", Kind::Uint, 64}, {"int8", Kind::Int, 8},     {"int16", Kind::Int, 16},
                                {"int32", Kind::Int, 32},   {"int64", Kind::Int, 64},   {"float16", Kind::Float, 16},
                                {"float", Kind::Float, 32}, {"double", Kind::Float, 64}};
  for (const Entry& e : table)
    if (s == e.name) return t.kind = e.kind, t.bits = e.bits, true;
  return false;
}

bool parseDataType(const std::string& s, DataType& t) {
  size_t lt = s.find('<');
  if (lt == std::string::npos) return t.comps = t.rows = 1, parseScalarType(s, t);
  if (s.back() != '>' || !parseScalarType(s.substr(lt + 1, s.size() - lt - 2), t)) return false;
  std::string head = s.substr(0, lt);
  if (head.size() == 4 && head.compare(0, 3, "vec") == 0 && head[3] >= '2' && head[3] <= '4')
    return t.comps = t.rows = unsigned(head[3] - '0'), true;
  if (head.size() == 6 && head.compare(0, 3, "mat") == 0 && head[4] == 'x' && head[3] >= '2' && head[3] <= '4' &&
      head[5] >= '2' && head[5] <= '4')
    return t.rows = unsigned(head[5] - '0'), t.comps = unsigned((head[3] - '0') * t.rows), true;
  return false;
}

uint16_t floatToHalf(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u, m = x & 0x7fffffu;
  int32_t e = int32_t((x >> 23) & 0xff);
  if (e == 0xff) return uint16_t(sign | 0x7c00u | (m ? 0x200u | (m >> 13) : 0u));
  int32_t ne = e - 127 + 15;
  if (ne >= 31) return uint16_t(sign | 0x7c00u);
  if (ne <= 0) {
    if (ne < -10) return uint16_t(sign);
    m |= 0x800000u;
    uint32_t shift = uint32_t(14 - ne), half = m >> shift, rem = m & ((1u << shift) - 1), mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) ++half;
    return uint16_t(sign | half);
  }
  uint32_t half = (uint32_t(ne) << 10) | (m >> 13), rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) ++half;
  return uint16_t(sign | half);
}

float halfToFloat(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu, x;
  if (e == 0) {
    if (m == 0) {
      x = sign;
    } else {
      e = 127 - 15 + 1;
      while (!(m & 0x400u)) m <<= 1, --e;
      x = sign | (e << 23) | ((m & 0x3ffu) << 13);
    }
  } else if (e == 31) {
    x = sign | 0x7f800000u | (m << 13);
  } else {
    x = sign | ((e + 127 - 15) << 23) | (m << 13);
  }
  float f;
  memcpy(&f, &x, 4);
  return f;
}

// Appends one scalar of type t spelled as token to out. Returns false when the token is not a number of that kind.
bool encodeScalar(const DataType& t, const std::string& token, std::vector<uint8_t>& out) {
  uint64_t raw = 0;
  char* end = nullptr;
  const char* s = token.c_str();
  bool hex = token.size() > 2 && token[0] == '0' && (token[1] == 'x' || token[1] == 'X');
  if (t.kind == Kind::Float && !hex) {
    double d = strtod(s, &end);
    if (end == s || *end) return false;
    if (t.bits == 64) {
      memcpy(&raw, &d, 8);
    } else if (t.bits == 32) {
      float f = float(d);
      uint32_t b;
      memcpy(&b, &f, 4);
      raw = b;
    } else {
      raw = floatToHalf(float(d));
    }
  } else if (t.kind == Kind::Int && !hex) {
    long long v = strtoll(s, &end, 10);
    if (end == s || *end) {  // "0.0" is accepted for integer buffers
      double d = strtod(s, &end);
      if (end == s || *end) return false;
      v = (long long)d;
    }
    raw = uint64_t(v);
  } else {
    raw = strtoull(s, &end, hex ? 16 : 10);
    if (end == s || *end) {  // an int token such as -1 for a uint buffer
      long long v = strtoll(s, &end, 10);
      if (end == s || *end) {
        double d = strtod(s, &end);
        if (end == s || *end) return false;
        v = (long long)d;
      }
      raw = uint64_t(v);
    }
  }
  for (unsigned i = 0; i < t.scalarBytes(); ++i) out.push_back(uint8_t(raw >> (8 * i)));
  return true;
}

uint64_t loadBits(const DataType& t, const uint8_t* p) {
  uint64_t v = 0;
  memcpy(&v, p, t.scalarBytes());  // little endian
  return v;
}

double scalarToDouble(const DataType& t, const uint8_t* p) {
  uint64_t bits = loadBits(t, p);
  switch (t.kind) {
    case Kind::Float:
      if (t.bits == 64) { double d; memcpy(&d, &bits, 8); return d; }
      if (t.bits == 32) { uint32_t b = uint32_t(bits); float f; memcpy(&f, &b, 4); return f; }
      return halfToFloat(uint16_t(bits));
    case Kind::Int: {
      int64_t v = int64_t(bits << (64 - t.bits)) >> (64 - t.bits);
      return double(v);
    }
    case Kind::Uint: return double(bits);
  }
  return 0;
}

std::string showScalar(const DataType& t, const uint8_t* p) {
  uint64_t bits = loadBits(t, p);
  if (t.kind == Kind::Float) return format("%.9g", scalarToDouble(t, p)) + format(" (0x%0*llx)", int(t.bits / 4), (unsigned long long)bits);
  if (t.kind == Kind::Int) return format("%lld (0x%0*llx)", (long long)int64_t(bits << (64 - t.bits)) >> (64 - t.bits), int(t.bits / 4), (unsigned long long)bits);
  return format("%llu (0x%0*llx)", (unsigned long long)bits, int(t.bits / 4), (unsigned long long)bits);
}

// ---------------------------------------------------------------------------------------------------------------
// Amber document
// ---------------------------------------------------------------------------------------------------------------

struct Shader {
  std::string stage, name, format, env, source;
};

struct Buffer {
  std::string name;
  DataType type;
  bool typeOk = true;
  std::vector<uint8_t> bytes;
};

struct Bind {
  std::string buffer, kind;
  bool hasSlot = false;
  uint32_t set = 0, binding = 0;
};

struct Pipeline {
  std::string name, type;
  std::vector<std::string> attach;
  std::vector<Bind> binds;
};

struct Command {
  enum Op { Run, Expect } op = Run;
  std::vector<std::string> words;  // tokens of the statement
  unsigned x = 0, y = 0, z = 0;    // Run
  bool hasType = false;            // Expect from a VkScript probe: element type of the probe, not of the buffer
  DataType type;
};

struct Document {
  std::map<std::string, Shader> shaders;
  std::map<std::string, Buffer> buffers;
  std::map<std::string, Pipeline> pipelines;
  std::vector<Command> commands;
  std::vector<std::string> unsupported;  // distinct Amber features the runner lacks
  void missing(const std::string& what) {
    if (std::find(unsupported.begin(), unsupported.end(), what) == unsupported.end()) unsupported.push_back(what);
  }
};

bool parseUnsigned(const std::string& s, unsigned& v) {
  char* end = nullptr;
  unsigned long x = strtoul(s.c_str(), &end, 0);
  if (s.empty() || *end) return false;
  return v = unsigned(x), true;
}

void parseBuffer(Document& doc, const fs::path& dir, const std::vector<std::string>& lines, size_t& i, std::vector<std::string> t) {
  Buffer b;
  b.name = t.at(1);
  size_t size = 0;
  bool haveSize = false, fill = false, series = false, haveData = false;
  std::string fillValue, seriesFrom, seriesInc;
  std::vector<std::string> data;
  for (size_t j = 2; j < t.size(); ++j) {
    const std::string& w = t[j];
    auto arg = [&]() -> std::string { return j + 1 < t.size() ? t[++j] : std::string(); };
    if (w == "DATA_TYPE") {
      std::string ty = arg();
      if (!parseDataType(ty, b.type)) b.typeOk = false, doc.missing("BUFFER DATA_TYPE " + ty);
    } else if (w == "FORMAT") {
      b.typeOk = false;
      doc.missing("BUFFER FORMAT " + arg());
    } else if (w == "SIZE") {
      haveSize = true, size = strtoull(arg().c_str(), nullptr, 0);
    } else if (w == "FILL") {
      fill = true, fillValue = arg();
    } else if (w == "SERIES_FROM") {
      series = true, seriesFrom = arg();
    } else if (w == "INC_BY") {
      seriesInc = arg();
    } else if (w == "FILE") {
      std::string kind = arg(), name = arg();
      if (kind != "TEXT") {
        doc.missing("BUFFER FILE " + kind);
      } else {
        haveData = true;
        for (const std::string& line : splitLines(readFile((dir / name).string())))
          for (const std::string& x : splitWords(stripComment(line))) data.push_back(x);
      }
    } else if (w == "STD140" || w == "STD430") {
      // Layout keywords do not change the packed element encoding used here.
    } else if (w == "DATA") {
      haveData = true;
      bool ended = false;
      for (++j; j < t.size(); ++j) {
        if (t[j] == "END") { ended = true; break; }
        data.push_back(t[j]);
      }
      while (!ended && i + 1 < lines.size()) {
        auto lw = splitWords(stripComment(lines[++i]));
        for (const std::string& x : lw) {
          if (x == "END") { ended = true; break; }
          data.push_back(x);
        }
      }
      break;
    } else {
      doc.missing("BUFFER " + w);
    }
  }
  if (b.typeOk) {
    if (haveData) {
      for (const std::string& v : data)
        if (!encodeScalar(b.type, v, b.bytes)) { doc.missing("BUFFER DATA value " + v); break; }
      padLayout(b.type, b.bytes);
      if (haveSize) b.bytes.resize(std::max<size_t>(b.bytes.size(), size * b.type.elementBytes()));
    } else if (haveSize) {
      b.bytes.reserve(size * b.type.elementBytes());
      for (size_t e = 0; e < size * b.type.comps; ++e) {
        std::string tok = fillValue;
        if (series) {
          // SERIES_FROM start INC_BY step, per scalar
          double a = strtod(seriesFrom.c_str(), nullptr), s = strtod(seriesInc.c_str(), nullptr);
          double v = a + s * double(e);
          tok = b.type.kind == Kind::Float ? format("%.17g", v) : format("%lld", (long long)v);
        } else if (!fill) {
          tok = "0";
        }
        if (!encodeScalar(b.type, tok, b.bytes)) { doc.missing("BUFFER FILL value " + tok); break; }
      }
      padLayout(b.type, b.bytes);
    }
  }
  doc.buffers[b.name] = std::move(b);
}

void parsePipeline(Document& doc, const std::vector<std::string>& lines, size_t& i, const std::vector<std::string>& t) {
  Pipeline p;
  p.type = t.at(1);
  p.name = t.at(2);
  for (++i; i < lines.size(); ++i) {
    auto w = splitWords(stripComment(lines[i]));
    if (w.empty()) continue;
    if (w[0] == "END") break;
    if (p.type != "compute") continue;
    if (w[0] == "ATTACH" && w.size() >= 2) {
      p.attach.push_back(w[1]);
      for (size_t k = 2; k < w.size(); ++k)
        if (w[k] != "ENTRY_POINT" && k != 3) doc.missing("ATTACH " + w[k]);
    } else if (w[0] == "BIND" && w.size() >= 5 && w[1] == "BUFFER" && w[3] == "AS") {
      Bind b;
      b.buffer = w[2];
      b.kind = w[4];
      for (size_t k = 5; k < w.size(); ++k) {
        unsigned v;
        if (w[k] == "DESCRIPTOR_SET" && k + 1 < w.size() && parseUnsigned(w[k + 1], v)) b.set = v, b.hasSlot = true, ++k;
        else if (w[k] == "BINDING" && k + 1 < w.size() && parseUnsigned(w[k + 1], v)) b.binding = v, ++k;
        else doc.missing("BIND " + w[k]);
      }
      p.binds.push_back(b);
    } else if (w[0] == "SUBGROUP") {
      // Subgroup size control requests (a nested block ended by its own END) do not apply to the Metal dispatch.
      for (++i; i < lines.size() && splitWords(stripComment(lines[i])) != std::vector<std::string>{"END"}; ++i) {
      }
    } else {
      doc.missing(w[0] == "BIND" && w.size() > 1 ? "BIND " + w[1] : "PIPELINE compute " + w[0]);
    }
  }
  doc.pipelines[p.name] = std::move(p);
}


// Legacy Amber "VkScript" form ([compute shader spirv], [test] with ssbo / uniform ubo / compute / probe ssbo). It is
// converted into the same Document: buffers "ssbo S:B" / "ubo S:B", one compute pipeline "main", RUN and EXPECT commands.
bool isVkScript(const std::string& text) {
  for (const std::string& l : splitLines(text))
    if (l.compare(0, 8, "[compute") == 0 || l == "[test]") return true;
  return false;
}

bool parseVkType(const std::string& s, DataType& t) {
  std::string base = s;
  unsigned comps = 1;
  if (s.size() > 3 && s.compare(s.size() - 1, 1, "2") >= 0 && s.compare(s.size() - 1, 1, "4") <= 0 &&
      (s.compare(0, 3, "vec") == 0 || s.compare(0, 4, "ivec") == 0 || s.compare(0, 4, "uvec") == 0)) {
    comps = unsigned(s.back() - '0');
    base = s[0] == 'i' ? "int" : s[0] == 'u' ? "uint" : "float";
  }
  struct Entry { const char* name; Kind kind; unsigned bits; };
  static const Entry table[] = {{"int", Kind::Int, 32}, {"uint", Kind::Uint, 32}, {"float", Kind::Float, 32},
                                {"double", Kind::Float, 64}, {"int64_t", Kind::Int, 64}, {"uint64_t", Kind::Uint, 64},
                                {"int16_t", Kind::Int, 16}, {"uint16_t", Kind::Uint, 16}, {"int8_t", Kind::Int, 8},
                                {"uint8_t", Kind::Uint, 8}, {"float16_t", Kind::Float, 16}};
  for (const Entry& e : table)
    if (base == e.name) return t.kind = e.kind, t.bits = e.bits, t.comps = t.rows = comps, true;
  return false;
}

Document parseVkScript(const std::string& text) {
  Document doc;
  std::string section, env;
  std::map<std::string, std::string> shaderText;
  std::string shaderFormat;
  Pipeline pipe;
  pipe.name = "main", pipe.type = "compute", pipe.attach = {"cs"};
  std::set<std::string> bound;
  bool ran = false;
  auto buffer = [&](const std::string& kind, const std::string& slot) -> Buffer& {
    std::string name = kind + " " + slot;
    Buffer& b = doc.buffers[name];
    if (b.name.empty()) {
      b.name = name;
      b.type.kind = Kind::Uint, b.type.bits = 32, b.type.comps = 1;
      Bind bind;
      bind.buffer = name, bind.kind = kind == "ubo" ? "uniform" : "storage", bind.hasSlot = true;
      size_t colon = slot.find(':');
      // "S:B" is set and binding; a lone number is a binding in set 0.
      bind.set = colon == std::string::npos ? 0 : unsigned(atoi(slot.c_str()));
      bind.binding = unsigned(atoi(colon == std::string::npos ? slot.c_str() : slot.c_str() + colon + 1));
      pipe.binds.push_back(bind);
    }
    return b;
  };
  for (const std::string& raw : splitLines(text)) {
    std::string line = trim(raw);
    if (!line.empty() && line[0] == '[') {
      section = line;
      continue;
    }
    if (section == "[require]") {
      if (line.find("spirv_1_4") != std::string::npos) env = "spv1.4";
      continue;
    }
    if (section == "[compute shader spirv]" || section == "[compute shader]") {
      shaderText[section] += raw + "\n";
      continue;
    }
    line = trim(stripComment(raw));
    if (line.empty()) continue;
    if (section != "[test]") {
      doc.missing("VkScript section " + section);
      continue;
    }
    std::vector<std::string> t = splitWords(line);
    if (t[0] == "compute" && t.size() == 4 && t[1] != "entrypoint") {
      Command c;
      c.op = Command::Run;
      c.words = {"RUN", "main"};
      parseUnsigned(t[1], c.x), parseUnsigned(t[2], c.y), parseUnsigned(t[3], c.z);
      doc.commands.push_back(c);
      ran = true;
    } else if (t[0] == "compute" && t.size() == 3 && t[1] == "entrypoint") {
      // The translated module has a single entry point named main.
    } else if ((t[0] == "ssbo" || (t[0] == "uniform" && t.size() > 1 && t[1] == "ubo")) && t.size() >= 3) {
      size_t at = t[0] == "ssbo" ? 1 : 2;
      std::string kind = t[0] == "ssbo" ? "ssbo" : "ubo";
      if (ran) { doc.missing("VkScript buffer write after compute"); continue; }
      Buffer& b = buffer(kind, t[at]);
      size_t k = at + 1;
      DataType ty;
      if (k < t.size() && t[k] == "subdata") ++k;
      if (k < t.size() && parseVkType(t[k], ty)) {
        unsigned off = 0;
        if (k + 1 >= t.size() || !parseUnsigned(t[k + 1], off)) { doc.missing("VkScript " + t[0] + " form"); continue; }
        std::vector<uint8_t> bytes;
        for (size_t v = k + 2; v < t.size(); ++v)
          if (!encodeScalar(ty, t[v], bytes)) { doc.missing("VkScript value " + t[v]); break; }
        padLayout(ty, bytes);
        if (b.bytes.size() < off + bytes.size()) b.bytes.resize(off + bytes.size());
        memcpy(b.bytes.data() + off, bytes.data(), bytes.size());
      } else {
        unsigned size = 0;
        if (k < t.size() && parseUnsigned(t[k], size)) b.bytes.resize(std::max<size_t>(b.bytes.size(), size));
        else doc.missing("VkScript " + t[0] + " " + (k < t.size() ? t[k] : std::string("form")));
      }
    } else if (t[0] == "probe" && t.size() >= 7 && t[1] == "ssbo") {
      Command c;
      c.op = Command::Expect;
      if (!parseVkType(t[2], c.type)) { doc.missing("VkScript probe type " + t[2]); continue; }
      c.hasType = true;
      std::string name = "ssbo " + t[3];
      if (!doc.buffers.count(name)) { doc.missing("VkScript probe of unwritten " + name); continue; }
      c.words = {"EXPECT", name, "IDX", t[4]};
      if (t[5] == "~=") c.words.insert(c.words.end(), {"TOLERANCE", "0.01"});
      else if (t[5] != "==") { doc.missing("VkScript probe " + t[5]); continue; }
      c.words.push_back("EQ");
      c.words.insert(c.words.end(), t.begin() + 6, t.end());
      doc.commands.push_back(c);
    } else {
      doc.missing("VkScript " + t[0] + (t.size() > 1 ? " " + t[1] : std::string()));
    }
  }
  for (auto& kv : shaderText) {
    Shader s;
    s.stage = "compute", s.name = "cs", s.format = kv.first == "[compute shader]" ? "GLSL" : "SPIRV-ASM";
    s.env = env, s.source = kv.second;
    doc.shaders["cs"] = s;
  }
  doc.pipelines["main"] = pipe;
  return doc;
}

Document parseDocument(const std::string& text, const fs::path& dir) {
  if (isVkScript(text)) return parseVkScript(text);
  Document doc;
  std::vector<std::string> lines = splitLines(text);
  for (size_t i = 0; i < lines.size(); ++i) {
    std::vector<std::string> t = splitWords(stripComment(lines[i]));
    if (t.empty()) continue;
    const std::string& kw = t[0];
    if (kw == "SHADER") {
      if (t.size() < 4) { doc.missing("SHADER form"); continue; }
      Shader s;
      s.stage = t[1], s.name = t[2], s.format = t[3];
      for (size_t k = 4; k + 1 < t.size(); ++k)
        if (t[k] == "TARGET_ENV") s.env = t[k + 1];
      if (s.format != "PASSTHROUGH") {
        std::string body;
        for (++i; i < lines.size() && trim(lines[i]) != "END"; ++i) body += lines[i] + "\n";
        s.source = body;
      }
      doc.shaders[s.name] = std::move(s);
    } else if (kw == "BUFFER") {
      if (t.size() < 2) { doc.missing("BUFFER form"); continue; }
      parseBuffer(doc, dir, lines, i, t);
    } else if (kw == "PIPELINE") {
      if (t.size() < 3) { doc.missing("PIPELINE form"); continue; }
      parsePipeline(doc, lines, i, t);
    } else if (kw == "RUN") {
      Command c;
      c.op = Command::Run;
      c.words = t;
      if (t.size() == 5 && parseUnsigned(t[2], c.x) && parseUnsigned(t[3], c.y) && parseUnsigned(t[4], c.z)) {
        doc.commands.push_back(c);
      } else {
        doc.missing(t.size() >= 3 ? "RUN " + t[2] : "RUN form");
      }
    } else if (kw == "EXPECT") {
      Command c;
      c.op = Command::Expect;
      c.words = t;
      doc.commands.push_back(c);
    } else if (kw == "DEVICE_FEATURE" || kw == "DEVICE_EXTENSION" || kw == "INSTANCE_EXTENSION" ||
               kw == "DEVICE_PROPERTY" || kw == "SET") {
      // Capability requirements are informational here: Metal either runs the shader or the result shows why not.
    } else if (kw == "END") {
      // stray terminator after a one-line block
    } else {
      doc.missing(kw);
    }
  }
  return doc;
}

// ---------------------------------------------------------------------------------------------------------------
// Shader front end
// ---------------------------------------------------------------------------------------------------------------

struct Tools {
  std::string glslang = "/opt/homebrew/bin/glslangValidator", spirvAs = "/opt/homebrew/bin/spirv-as",
              spirvOpt = "/opt/homebrew/bin/spirv-opt";
  bool inlineFunctions = false;  // --inline: run spirv-opt to inline calls and drop dead functions before translating
  fs::path tmp;
};

// Runs argv with stdout+stderr redirected to logPath. Returns the exit status, or -1 when it could not run.
int runTool(const std::vector<std::string>& args, const fs::path& logPath) {
  std::vector<char*> argv;
  for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 1, logPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  posix_spawn_file_actions_adddup2(&fa, 1, 2);
  pid_t pid;
  int rc = posix_spawn(&pid, args[0].c_str(), &fa, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  if (rc != 0) return -1;
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::string spvEnv(const std::string& env, bool glsl) {
  // TARGET_ENV spv1.x selects the SPIR-V version. Default: Vulkan 1.1 (SPIR-V 1.3).
  if (env.empty()) return "vulkan1.1";
  if (glsl) {
    if (env == "spv1.0") return "vulkan1.0";
    if (env == "spv1.4" || env == "spv1.5") return "vulkan1.2";
    if (env == "spv1.6") return "vulkan1.3";
    return "vulkan1.1";
  }
  return env;
}

std::vector<uint32_t> compileShader(const Tools& tools, const Shader& s, int serial) {
  fs::path base = tools.tmp / format("s%d", serial);
  fs::path spv = base.string() + ".spv", log = base.string() + ".log";
  int rc;
  if (s.format == "GLSL") {
    std::string stage = s.stage == "compute" ? "comp" : s.stage;
    fs::path src = base.string() + "." + stage;
    std::string text = s.source;
    if (text.find("#version") == std::string::npos) text = "#version 450\n" + text;
    std::ofstream(src) << text;
    rc = runTool({tools.glslang, "-V", "--target-env", spvEnv(s.env, true), "-S", stage, "-o", spv.string(), src.string()}, log);
  } else if (s.format == "SPIRV-ASM") {
    fs::path src = base.string() + ".spvasm";
    std::ofstream(src) << s.source;
    rc = runTool({tools.spirvAs, "--target-env", spvEnv(s.env, false), "-o", spv.string(), src.string()}, log);
  } else {
    throw Stop{Status::UnsupportedAmber, "SHADER format " + s.format};
  }
  if (rc != 0) {
    std::string out = trim(readFile(log.string()));
    size_t p = out.find("ERROR");
    throw Stop{Status::UnsupportedAmber, "shader front end (" + s.format + "): " + oneLine(p == std::string::npos ? out : out.substr(p), 300)};
  }
  if (tools.inlineFunctions) {
    fs::path opt = base.string() + ".opt.spv";
    if (runTool({tools.spirvOpt, "--inline-entry-points-exhaustive", "--eliminate-dead-functions", "--eliminate-local-single-store",
                 "--target-env=" + spvEnv(s.env, s.format == "GLSL"), spv.string(), "-o", opt.string()}, log) != 0)
      throw Stop{Status::UnsupportedAmber, "spirv-opt failed: " + oneLine(readFile(log.string()), 200)};
    spv = opt;
  }
  std::string bytes = readFile(spv.string());
  std::vector<uint32_t> words(bytes.size() / 4);
  memcpy(words.data(), bytes.data(), words.size() * 4);
  return words;
}

// LocalSize of the module's only entry point, scanning OpExecutionMode.
bool localSize(const std::vector<uint32_t>& w, uint32_t out[3]) {
  for (size_t i = 5; i < w.size();) {
    uint32_t op = w[i] & 0xffff, n = w[i] >> 16;
    if (n == 0) break;
    if (op == 16 && n >= 6 && w[i + 2] == 17) return out[0] = w[i + 3], out[1] = w[i + 4], out[2] = w[i + 5], true;
    i += n;
  }
  return false;
}

// ---------------------------------------------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------------------------------------------

struct Compiled {
  id<MTLComputePipelineState> pso = nil;
  uint32_t local[3] = {1, 1, 1};
  std::vector<std::pair<std::string, unsigned>> buffers;  // amber buffer -> Metal buffer index
  int pushIndex = -1;
  std::string pushBuffer;
};

Compiled build(const Tools& tools, id<MTLDevice> device, const Document& doc, const Pipeline& p, int serial,
               std::string& translateNote) {
  if (p.attach.size() != 1) throw Stop{Status::UnsupportedAmber, "pipeline with " + std::to_string(p.attach.size()) + " ATTACH"};
  auto it = doc.shaders.find(p.attach[0]);
  if (it == doc.shaders.end()) throw Stop{Status::UnsupportedAmber, "ATTACH of unknown shader " + p.attach[0]};
  if (it->second.stage != "compute") throw Stop{Status::UnsupportedAmber, "ATTACH of " + it->second.stage + " shader"};
  std::vector<uint32_t> words = compileShader(tools, it->second, serial);

  Compiled c;
  spirv2air::Options options;
  unsigned index = 0;
  for (const Bind& b : p.binds) {
    if (b.kind == "storage" || b.kind == "uniform") {
      if (!b.hasSlot) throw Stop{Status::UnsupportedAmber, "BIND without DESCRIPTOR_SET/BINDING"};
      spirv2air::ResourceBinding rb;
      rb.set = b.set, rb.binding = b.binding, rb.buffer = index, rb.texture = 0;
      options.bindings.push_back(rb);
      c.buffers.emplace_back(b.buffer, index++);
    } else if (b.kind != "push_constant") {
      translateNote += (translateNote.empty() ? "" : ", ") + std::string("BIND AS ") + b.kind;
    }
  }
  for (const Bind& b : p.binds)
    if (b.kind == "push_constant") c.pushIndex = int(index), c.pushBuffer = b.buffer, options.pushConstantBuffer = int(index++);
  if (!translateNote.empty()) return c;  // caller reports UNSUPPORTED_AMBER with the translation outcome appended

  spirv2air::Result r = spirv2air::translate(words.data(), words.size(), options);
  if (!r.ok()) throw Stop{Status::TranslateReject, oneLine(r.error)};
  if (!localSize(words, c.local)) throw Stop{Status::UnsupportedAmber, "module without literal LocalSize"};
  NSError* error = nil;
  dispatch_data_t data = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
  id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
  if (!library) throw Stop{Status::Fail, "metal newLibraryWithData: " + oneLine(error.description.UTF8String)};
  id<MTLFunction> function = [library newFunctionWithName:@"main"];
  if (!function || function.functionType != MTLFunctionTypeKernel) throw Stop{Status::Fail, "metal: no kernel main"};
  c.pso = [device newComputePipelineStateWithFunction:function error:&error];
  if (!c.pso) throw Stop{Status::Fail, "metal pipeline: " + oneLine(error.description.UTF8String)};
  NSUInteger need = NSUInteger(c.local[0]) * c.local[1] * c.local[2];
  if (c.pso.maxTotalThreadsPerThreadgroup < need)
    throw Stop{Status::Fail, format("metal pipeline allows %lu threads per group, shader needs %lu",
                                    (unsigned long)c.pso.maxTotalThreadsPerThreadgroup, (unsigned long)need)};
  return c;
}

struct Tolerance {
  double value = 0;
  bool relative = false;
};

struct Verdict {
  unsigned failures = 0;
  std::vector<std::string> notes;
  void add(const std::string& n) {
    if (++failures <= 3) notes.push_back(n);
  }
};

bool closeEnough(const DataType& t, const uint8_t* got, const uint8_t* want, const Tolerance& tol) {
  if (t.kind != Kind::Float) return loadBits(t, got) == loadBits(t, want);
  double a = scalarToDouble(t, got), b = scalarToDouble(t, want);
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  if (a == b) return true;
  if (std::isinf(a) || std::isinf(b)) return false;
  double allowed = tol.relative ? tol.value / 100.0 * std::fabs(b) : tol.value;
  return std::fabs(a - b) <= allowed;
}

void checkExpect(const Document& doc, std::map<std::string, id<MTLBuffer>>& live, const Command& cmd, Verdict& v) {
  const std::vector<std::string>& w = cmd.words;
  if (w.size() < 3) return v.add("EXPECT: malformed"), void();
  auto bit = doc.buffers.find(w[1]);
  if (bit == doc.buffers.end()) return v.add("EXPECT of unknown buffer " + w[1]), void();
  const Buffer& buf = bit->second;
  const uint8_t* mem = static_cast<const uint8_t*>(live[w[1]].contents);
  size_t memSize = buf.bytes.size();

  size_t k = 2;
  std::vector<unsigned> idx, size;
  if (w[k] == "IDX") {
    for (++k; k < w.size(); ++k) {
      unsigned x;
      if (!parseUnsigned(w[k], x)) break;
      idx.push_back(x);
    }
  }
  if (k < w.size() && w[k] == "SIZE") {
    for (++k; k < w.size(); ++k) {
      unsigned x;
      if (!parseUnsigned(w[k], x)) break;
      size.push_back(x);
    }
  }
  std::vector<Tolerance> tols;
  if (k < w.size() && w[k] == "TOLERANCE") {
    for (++k; k < w.size(); ++k) {
      if (w[k] == "EQ") break;
      char* end = nullptr;
      Tolerance t;
      t.value = strtod(w[k].c_str(), &end);
      if (end == w[k].c_str()) break;
      if (*end == '%') t.relative = true, ++end;
      if (*end) break;
      tols.push_back(t);
    }
  }
  if (k >= w.size()) return v.add("EXPECT " + buf.name + ": no comparator"), void();
  const std::string& op = w[k];
  if (idx.size() > 1 || size.size() > 1) return v.add("EXPECT " + buf.name + ": image region form unsupported"), void();
  size_t offset = idx.empty() ? 0 : idx[0];
  if (!buf.typeOk && !cmd.hasType) return v.add("EXPECT " + buf.name + ": buffer without a runner type"), void();
  const DataType& t = cmd.hasType ? cmd.type : buf.type;
  unsigned sb = t.scalarBytes();

  if (op == "EQ") {
    std::vector<uint8_t> want;
    for (++k; k < w.size(); ++k)
      if (!encodeScalar(t, w[k], want)) return v.add("EXPECT " + buf.name + ": bad value '" + w[k] + "'"), void();
    padLayout(t, want);
    size_t n = want.size() / sb;
    if (offset + want.size() > memSize)
      return v.add(format("EXPECT %s IDX %zu: %zu values run past the %zu byte buffer", buf.name.c_str(), offset, n, memSize)), void();
    unsigned bad = 0;
    std::string first;
    for (size_t i = 0; i < n; ++i) {
      if (t.isPad(i)) continue;
      Tolerance tol = tols.empty() ? Tolerance{1e-6, false} : tols[i % tols.size()];
      if (closeEnough(t, mem + offset + i * sb, want.data() + i * sb, tol)) continue;
      if (!bad++) first = format("IDX %zu: got %s want %s", offset + i * sb, showScalar(t, mem + offset + i * sb).c_str(), showScalar(t, want.data() + i * sb).c_str());
    }
    if (bad) v.add(format("EXPECT %s %s (%u of %zu values differ)", buf.name.c_str(), first.c_str(), bad, n));
  } else if (op == "EQ_BUFFER") {
    if (k + 1 >= w.size()) return v.add("EXPECT EQ_BUFFER without a buffer"), void();
    auto oit = doc.buffers.find(w[k + 1]);
    if (oit == doc.buffers.end()) return v.add("EXPECT EQ_BUFFER unknown buffer " + w[k + 1]), void();
    const uint8_t* other = static_cast<const uint8_t*>(live[w[k + 1]].contents);
    size_t otherSize = oit->second.bytes.size();
    if (memSize != otherSize)
      return v.add(format("EXPECT %s EQ_BUFFER %s: sizes differ (%zu vs %zu bytes)", buf.name.c_str(), w[k + 1].c_str(), memSize, otherSize)), void();
    if (!memcmp(mem, other, memSize)) return;
    Tolerance tol = tols.empty() ? Tolerance{1e-6, false} : tols[0];
    unsigned bad = 0;
    std::string first;
    for (size_t off = 0; off + sb <= memSize; off += sb) {
      if (t.isPad(off / sb)) continue;
      if (closeEnough(t, mem + off, other + off, tol)) continue;
      if (!bad++) first = format("IDX %zu: got %s want %s", off, showScalar(t, mem + off).c_str(), showScalar(t, other + off).c_str());
    }
    if (bad) v.add(format("EXPECT %s EQ_BUFFER %s %s (%u of %zu values differ)", buf.name.c_str(), w[k + 1].c_str(), first.c_str(), bad, memSize / sb));
  } else {
    v.add("EXPECT comparator " + op + " unsupported");
  }
}

Status execute(const Tools& tools, const Document& doc, std::string& detail) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) throw Stop{Status::Crash, "no Metal device"};
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Compile every pipeline the script runs, even when an Amber feature is missing: that tells whether spirv2air
    // accepts the shader independently of the runner's gaps.
    std::map<std::string, Compiled> compiled;
    std::string note;
    int serial = 0;
    std::string rejected;
    for (const Command& c : doc.commands) {
      if (c.op != Command::Run || compiled.count(c.words[1])) continue;
      auto pit = doc.pipelines.find(c.words[1]);
      if (pit == doc.pipelines.end()) throw Stop{Status::UnsupportedAmber, "RUN of unknown pipeline " + c.words[1]};
      if (pit->second.type != "compute") { throw Stop{Status::UnsupportedAmber, "RUN of " + pit->second.type + " pipeline (graphics)"}; }
      std::string bindNote;
      try {
        compiled[c.words[1]] = build(tools, device, doc, pit->second, serial++, bindNote);
      } catch (const Stop& s) {
        if (doc.unsupported.empty() && bindNote.empty()) throw;
        rejected = std::string(statusName(s.status)) + ": " + s.detail;
      }
      if (!bindNote.empty()) note += (note.empty() ? "" : ", ") + bindNote;
    }
    if (doc.commands.empty()) throw Stop{Status::UnsupportedAmber, "no RUN or EXPECT commands"};
    bool anyRun = false;
    for (const Command& c : doc.commands) anyRun |= c.op == Command::Run;
    if (!anyRun) throw Stop{Status::UnsupportedAmber, "no RUN command"};
    if (!doc.unsupported.empty() || !note.empty() || !rejected.empty()) {
      std::string d;
      for (const std::string& u : doc.unsupported) d += (d.empty() ? "" : ", ") + u;
      if (!note.empty()) d += (d.empty() ? "" : ", ") + note;
      if (!rejected.empty()) d += " [spirv2air: " + rejected + "]";
      else if (!note.empty()) d += " [spirv2air: translation skipped]";
      else d += " [spirv2air: translated ok]";
      throw Stop{Status::UnsupportedAmber, d};
    }

    std::map<std::string, id<MTLBuffer>> live;
    for (auto& kv : doc.buffers) {
      const Buffer& b = kv.second;
      NSUInteger length = std::max<size_t>(b.bytes.size(), 16);
      id<MTLBuffer> mb = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
      memset(mb.contents, 0, length);
      if (!b.bytes.empty()) memcpy(mb.contents, b.bytes.data(), b.bytes.size());
      live[kv.first] = mb;
    }

    Verdict verdict;
    for (const Command& cmd : doc.commands) {
      if (cmd.op == Command::Expect) {
        checkExpect(doc, live, cmd, verdict);
        continue;
      }
      const Compiled& c = compiled[cmd.words[1]];
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:c.pso];
      for (auto& b : c.buffers) {
        if (!live.count(b.first)) throw Stop{Status::UnsupportedAmber, "BIND of unknown buffer " + b.first};
        [e setBuffer:live[b.first] offset:0 atIndex:b.second];
      }
      if (c.pushIndex >= 0) {
        auto pb = doc.buffers.find(c.pushBuffer);
        if (pb == doc.buffers.end()) throw Stop{Status::UnsupportedAmber, "push constant buffer " + c.pushBuffer + " is undefined"};
        [e setBytes:pb->second.bytes.data() length:std::max<size_t>(pb->second.bytes.size(), 4) atIndex:NSUInteger(c.pushIndex)];
      }
      [e dispatchThreadgroups:MTLSizeMake(cmd.x, cmd.y, cmd.z) threadsPerThreadgroup:MTLSizeMake(c.local[0], c.local[1], c.local[2])];
      [e endEncoding];
      dispatch_semaphore_t done = dispatch_semaphore_create(0);
      [cb addCompletedHandler:^(id<MTLCommandBuffer>) { dispatch_semaphore_signal(done); }];
      [cb commit];
      if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 20 * NSEC_PER_SEC)) != 0)
        throw Stop{Status::Hang, "command buffer did not complete in 20 s"};
      if (cb.status != MTLCommandBufferStatusCompleted)
        throw Stop{Status::Fail, "command buffer failed: " + oneLine(cb.error.description.UTF8String)};
    }
    if (verdict.failures) {
      for (const std::string& n : verdict.notes) detail += (detail.empty() ? "" : " | ") + n;
      if (verdict.failures > verdict.notes.size()) detail += format(" | +%zu more failed EXPECTs", verdict.failures - verdict.notes.size());
      return Status::Fail;
    }
    return Status::Pass;
  }
}

// Child mode: one script, one RESULT line on stdout.
int runOne(const std::string& path, Tools tools) {
  Status status;
  std::string detail;
  try {
    std::error_code ec;
    tools.tmp = fs::path(tools.tmp.empty() ? fs::temp_directory_path() : tools.tmp) / format("amber.%d", getpid());
    fs::create_directories(tools.tmp, ec);
    std::string text = readFile(path);
    if (text.empty()) throw Stop{Status::UnsupportedAmber, "unreadable or empty script"};
    Document doc = parseDocument(text, fs::path(path).parent_path());
    status = execute(tools, doc, detail);
    fs::remove_all(tools.tmp, ec);
  } catch (const Stop& s) {
    status = s.status, detail = s.detail;
    std::error_code ec;
    fs::remove_all(tools.tmp, ec);
  }
  printf("RESULT\t%s\t%s\n", statusName(status), oneLine(detail).c_str());
  fflush(stdout);
  _exit(0);  // do not wait for Metal teardown
}

// ---------------------------------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------------------------------

struct Outcome {
  Status status;
  std::string detail;
  double seconds;
};

std::string selfPath() {
  char buf[4096];
  uint32_t n = sizeof buf;
  return _NSGetExecutablePath(buf, &n) == 0 ? buf : std::string();
}

Outcome runChild(const std::string& file, const std::string& tmp, int timeoutSeconds, bool inlineFunctions) {
  auto t0 = std::chrono::steady_clock::now();
  int fds[2];
  if (pipe(fds) != 0) return {Status::Crash, "pipe failed", 0};
  std::vector<std::string> args = {selfPath(), "--one", file};
  if (!tmp.empty()) args.push_back("--tmp"), args.push_back(tmp);
  if (inlineFunctions) args.push_back("--inline");
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
  posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
  posix_spawn_file_actions_addclose(&fa, fds[0]);
  posix_spawn_file_actions_addclose(&fa, fds[1]);
  pid_t pid;
  int rc = posix_spawn(&pid, args[0].c_str(), &fa, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  close(fds[1]);
  if (rc != 0) return close(fds[0]), Outcome{Status::Crash, "spawn failed", 0};
  std::string out;
  bool timedOut = false;
  for (;;) {
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (elapsed > timeoutSeconds) { timedOut = true; break; }
    pollfd p = {fds[0], POLLIN, 0};
    int pr = poll(&p, 1, 200);
    if (pr > 0) {
      char buf[4096];
      ssize_t n = read(fds[0], buf, sizeof buf);
      if (n <= 0) break;
      out.append(buf, size_t(n));
      if (out.size() > (1u << 20)) out.erase(0, out.size() - (1u << 19));
    }
  }
  if (timedOut) kill(pid, SIGKILL);
  close(fds[0]);
  int st = 0;
  waitpid(pid, &st, 0);
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (timedOut) return {Status::Hang, format("no result within %d s", timeoutSeconds), secs};
  std::string tail;
  for (const std::string& l : splitLines(out)) {
    if (l.compare(0, 7, "RESULT\t") == 0) {
      size_t a = l.find('\t', 7);
      std::string name = l.substr(7, a == std::string::npos ? std::string::npos : a - 7);
      std::string detail = a == std::string::npos ? "" : l.substr(a + 1);
      for (Status s : {Status::Pass, Status::Fail, Status::TranslateReject, Status::UnsupportedAmber, Status::Crash, Status::Hang})
        if (name == statusName(s)) return {s, detail, secs};
    }
    if (!trim(l).empty()) tail = l;
  }
  if (WIFSIGNALED(st)) return {Status::Crash, format("signal %d (%s); last output: %s", WTERMSIG(st), strsignal(WTERMSIG(st)), oneLine(tail, 200).c_str()), secs};
  return {Status::Crash, format("exit %d without a result; last output: %s", WEXITSTATUS(st), oneLine(tail, 200).c_str()), secs};
}

std::string csvField(const std::string& s) {
  std::string o = "\"";
  for (char c : s) o += c == '"' ? "\"\"" : std::string(1, c);
  return o + "\"";
}

bool isComputeOnly(const std::string& path) {
  std::string t = readFile(path);
  if (isVkScript(t)) return t.find("\n[compute shader") != std::string::npos && t.find("\n[vertex") == std::string::npos &&
                            t.find("\n[fragment") == std::string::npos;
  return t.find("PIPELINE compute") != std::string::npos && t.find("PIPELINE graphics") == std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
  std::string cts, csv, listFile, one;
  std::string tmp;
  int timeout = 30;
  size_t offset = 0, limit = SIZE_MAX;
  bool allCompute = false;
  Tools tools;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
    if (a == "--cts") cts = next();
    else if (a == "--csv") csv = next();
    else if (a == "--list") listFile = next();
    else if (a == "--one") one = next();
    else if (a == "--tmp") tmp = next();
    else if (a == "--timeout") timeout = atoi(next().c_str());
    else if (a == "--offset") offset = strtoull(next().c_str(), nullptr, 10);
    else if (a == "--limit") limit = strtoull(next().c_str(), nullptr, 10);
    else if (a == "--all-compute") allCompute = true;
    else if (a == "--inline") tools.inlineFunctions = true;
    else if (a == "--glslang") tools.glslang = next();
    else if (a == "--spirv-as") tools.spirvAs = next();
    else if (a == "--help" || a == "-h") {
      fprintf(stderr, "usage: amber_runner --cts DIR [--all-compute | --list FILE | FILE.amber...] [--csv OUT] [--timeout SEC] [--offset N] [--limit N] [--tmp DIR] [--inline]\n");
      return 0;
    } else files.push_back(a);
  }
  if (!tmp.empty()) tools.tmp = tmp;
  if (!one.empty()) return runOne(one, tools);

  if (cts.empty()) return fprintf(stderr, "amber_runner: --cts DIR is required\n"), 2;
  if (fs::exists(fs::path(cts) / "external/vulkancts/data/vulkan/amber")) cts = (fs::path(cts) / "external/vulkancts/data/vulkan/amber").string();
  if (!listFile.empty())
    for (const std::string& l : splitLines(readFile(listFile)))
      if (!trim(l).empty()) files.push_back(trim(l));
  if (allCompute) {
    std::vector<std::string> found;
    for (auto& e : fs::recursive_directory_iterator(cts))
      if (e.is_regular_file() && e.path().extension() == ".amber" && isComputeOnly(e.path().string()))
        found.push_back(fs::relative(e.path(), cts).string());
    std::sort(found.begin(), found.end());
    files.insert(files.end(), found.begin(), found.end());
  }
  if (offset > files.size()) offset = files.size();
  files = std::vector<std::string>(files.begin() + long(offset), files.begin() + long(offset + std::min(limit, files.size() - offset)));

  FILE* out = nullptr;
  if (!csv.empty()) {
    bool fresh = !fs::exists(csv) || fs::file_size(csv) == 0;
    out = fopen(csv.c_str(), "a");
    if (!out) return fprintf(stderr, "amber_runner: cannot open %s\n", csv.c_str()), 2;
    if (fresh) fputs("test,status,seconds,detail\n", out);
  }
  std::map<std::string, int> counts;
  for (const std::string& f : files) {
    std::string path = fs::path(f).is_absolute() ? f : (fs::path(cts) / f).string();
    std::string name = fs::path(f).is_absolute() ? fs::relative(f, cts).string() : fs::path(f).lexically_normal().string();
    Outcome o = runChild(path, tmp, timeout, tools.inlineFunctions);
    ++counts[statusName(o.status)];
    printf("%-18s %6.2fs %s  %s\n", statusName(o.status), o.seconds, name.c_str(), o.detail.substr(0, 160).c_str());
    fflush(stdout);
    if (out) {
      fprintf(out, "%s,%s,%.2f,%s\n", csvField(name).c_str(), statusName(o.status), o.seconds, csvField(o.detail).c_str());
      fflush(out);
    }
  }
  if (out) fclose(out);
  printf("-- %zu scripts:", files.size());
  for (auto& kv : counts) printf(" %s=%d", kv.first.c_str(), kv.second);
  printf("\n");
  return 0;
}
