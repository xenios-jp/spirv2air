#include "metallib.h"

#include <CommonCrypto/CommonDigest.h>

#include <cstdio>
#include <cstring>

namespace mtlb {
namespace {

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "metallib fields are written in host order");

constexpr size_t kTagHeader = 4 + 2;  // code + u16 length
constexpr size_t kEndt = 4;
// u32 size + NAME(len+1) + TYPE(1) + HASH(32) + MDSZ(8) + OFFT(24) + VERS(8) + ENDT
constexpr size_t kEntryFixed = 4 + kTagHeader * 6 + 1 + 1 + 32 + 8 + 24 + 8 + kEndt;

size_t entry_size(const Function& f) { return kEntryFixed + f.name.size() + (f.tessellation ? kTagHeader + 1 : 0); }

size_t cnst_payload(const Function& f) {
  size_t s = 2;
  for (uint32_t i = 0; i < f.num_constants; i++) s += f.constants[i].name.size() + 1 + 1 + 2 + 1;
  return s;
}

size_t vatt_payload(const Function& f) {
  size_t s = 2;
  for (uint32_t i = 0; i < f.num_attributes; i++) s += f.attributes[i].name.size() + 1 + 2;
  return s;
}

// Bytes of public-metadata tags for f, including ENDT (this is the block's u32 prefix value).
size_t public_tags_size(const Function& f) {
  size_t s = kEndt;
  if (f.num_constants) s += kTagHeader + cnst_payload(f);
  if (f.num_attributes) s += 2 * kTagHeader + vatt_payload(f) + 2 + f.num_attributes;
  return s;
}

struct Cursor {
  uint8_t* p;
  void bytes(const void* src, size_t n) {
    memcpy(p, src, n);
    p += n;
  }
  void u8(uint8_t v) { *p++ = v; }
  void u16(uint16_t v) { bytes(&v, 2); }
  void u32(uint32_t v) { bytes(&v, 4); }
  void u64(uint64_t v) { bytes(&v, 8); }
  void tag(const char (&code)[5], uint16_t len) {
    bytes(code, 4);
    u16(len);
  }
  void endt() { bytes("ENDT", 4); }
  void cstr(std::string_view s) {
    bytes(s.data(), s.size());
    u8(0);
  }
};

struct Layout {
  size_t funclist_entries, pub_size, priv_size, bitcode_size;
  size_t funclist_off, pub_off, priv_off, bitcode_off;
};

Layout layout(const Function* fns, size_t n) {
  Layout l{};
  for (size_t i = 0; i < n; i++) {
    l.funclist_entries += entry_size(fns[i]);
    l.pub_size += 4 + public_tags_size(fns[i]);
    l.priv_size += 4 + kEndt;
    l.bitcode_size += fns[i].bitcode_size;
  }
  l.funclist_off = kHeaderSize;
  l.pub_off = l.funclist_off + 4 + l.funclist_entries + kEndt;  // count, entries, header-extension ENDT
  l.priv_off = l.pub_off + l.pub_size;
  l.bitcode_off = l.priv_off + l.priv_size;
  return l;
}

}  // namespace

size_t prefix_size(const Function* fns, size_t n) { return layout(fns, n).bitcode_off; }

size_t library_size(const Function* fns, size_t n) {
  Layout l = layout(fns, n);
  return l.bitcode_off + l.bitcode_size;
}

Version air_version(Platform platform, Version os) {
  if (platform != Platform::MacOS) return {2, uint16_t(os.major >= 26 ? 8 : 7)};
  return {2, uint16_t(os.major >= 26 ? 8 : os.major >= 15 ? 7 : 6)};
}

Target make_target(Platform platform, Version min_os) {
  const uint16_t floor = platform == Platform::MacOS ? 14 : 18;
  if (min_os.major < floor) min_os = {floor, 0};
  Target t;
  t.platform = platform == Platform::MacOS ? 0x8001 : 0x0001;
  t.os = platform == Platform::MacOS ? 0x81 : platform == Platform::IOS ? 0x82 : 0x87;
  t.os_version = min_os;
  t.file_version = {2, uint16_t(air_version(platform, min_os).minor + 1)};
  return t;
}

const char* triple(Platform platform, Version os, char* buf, size_t size) {
  snprintf(buf, size, "air64_v2%u-apple-%s%u.%u.0%s", air_version(platform, os).minor,
           platform == Platform::MacOS ? "macosx" : "ios", os.major, os.minor,
           platform == Platform::IOSSimulator ? "-simulator" : "");
  return buf;
}

size_t write(uint8_t* out, const Target& t, const Function* fns, size_t n) {
  const Layout l = layout(fns, n);
  const size_t total = l.bitcode_off + l.bitcode_size;

  // Bitcode first: the zero-copy path already has it in place, so this is usually a no-op.
  uint8_t* bc = out + l.bitcode_off;
  for (size_t i = 0; i < n; i++) {
    if (fns[i].bitcode != bc) memmove(bc, fns[i].bitcode, fns[i].bitcode_size);
    bc += fns[i].bitcode_size;
  }

  Cursor c{out};
  c.bytes("MTLB", 4);
  c.u16(t.platform);
  c.u16(t.file_version.major);
  c.u16(t.file_version.minor);
  c.u8(t.file_type);
  c.u8(t.os);
  c.u16(t.os_version.major);
  c.u16(t.os_version.minor);
  c.u64(total);
  c.u64(l.funclist_off);
  c.u64(l.funclist_entries);
  c.u64(l.pub_off);
  c.u64(l.pub_size);
  c.u64(l.priv_off);
  c.u64(l.priv_size);
  c.u64(l.bitcode_off);
  c.u64(l.bitcode_size);

  // Function list.
  c.u32(static_cast<uint32_t>(n));
  uint64_t pub_rel = 0, priv_rel = 0, bc_rel = 0;
  for (size_t i = 0; i < n; i++) {
    const Function& f = fns[i];
    c.u32(static_cast<uint32_t>(entry_size(f)));
    c.tag("NAME", static_cast<uint16_t>(f.name.size() + 1));
    c.cstr(f.name);
    c.tag("TYPE", 1);
    c.u8(static_cast<uint8_t>(f.type));
    c.tag("HASH", 32);
    if (f.sha256) {
      c.bytes(f.sha256, 32);
    } else {
      CC_SHA256(out + l.bitcode_off + bc_rel, static_cast<CC_LONG>(f.bitcode_size), c.p);
      c.p += 32;
    }
    c.tag("MDSZ", 8);
    c.u64(f.bitcode_size);
    c.tag("OFFT", 24);
    c.u64(pub_rel);
    c.u64(priv_rel);
    c.u64(bc_rel);
    c.tag("VERS", 8);
    c.u16(f.air.major);
    c.u16(f.air.minor);
    c.u16(f.language.major);
    c.u16(f.language.minor);
    if (f.tessellation) {
      c.tag("TESS", 1);
      c.u8(f.tessellation);
    }
    c.endt();
    pub_rel += 4 + public_tags_size(f);
    priv_rel += 4 + kEndt;
    bc_rel += f.bitcode_size;
  }
  c.endt();  // header extension: none

  // Public metadata.
  for (size_t i = 0; i < n; i++) {
    const Function& f = fns[i];
    c.u32(static_cast<uint32_t>(public_tags_size(f)));
    if (f.num_constants) {
      c.tag("CNST", static_cast<uint16_t>(cnst_payload(f)));
      c.u16(static_cast<uint16_t>(f.num_constants));
      for (uint32_t k = 0; k < f.num_constants; k++) {
        const FunctionConstant& fc = f.constants[k];
        c.cstr(fc.name);
        c.u8(fc.type);
        c.u16(fc.index);
        c.u8(fc.required);
      }
    }
    if (f.num_attributes) {
      c.tag("VATT", static_cast<uint16_t>(vatt_payload(f)));
      c.u16(static_cast<uint16_t>(f.num_attributes));
      for (uint32_t k = 0; k < f.num_attributes; k++) {
        const VertexAttribute& a = f.attributes[k];
        c.cstr(a.name);
        c.u16(static_cast<uint16_t>(a.index | (a.used ? 0x8000 : 0)));
      }
      c.tag("VATY", static_cast<uint16_t>(2 + f.num_attributes));
      c.u16(static_cast<uint16_t>(f.num_attributes));
      for (uint32_t k = 0; k < f.num_attributes; k++) c.u8(f.attributes[k].type);
    }
    c.endt();
  }

  // Private metadata: always empty.
  for (size_t i = 0; i < n; i++) {
    c.u32(kEndt);
    c.endt();
  }
  return total;
}

}  // namespace mtlb
