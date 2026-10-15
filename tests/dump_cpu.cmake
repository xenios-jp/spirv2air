# Dumping intermediate IR must preserve the final shader, including dispatch-loop SSA numbering.
file(MAKE_DIRECTORY "${OUTPUT_DIR}/passes")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=S2A_DUMP_DIR --unset=S2A_STAGE_TIMES
  "${CLI}" "${INPUT}" "${OUTPUT_DIR}/plain.metallib" --buffer 0:0:0 --buffer 0:1:1
  RESULT_VARIABLE plain_result ERROR_VARIABLE plain_error)
if(NOT plain_result EQUAL 0)
  message(FATAL_ERROR "Undumped translation failed: ${plain_error}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=S2A_STAGE_TIMES "S2A_DUMP_DIR=${OUTPUT_DIR}/passes"
  "${CLI}" "${INPUT}" "${OUTPUT_DIR}/dumped.metallib" --buffer 0:0:0 --buffer 0:1:1
  RESULT_VARIABLE dumped_result ERROR_VARIABLE dumped_error)
if(NOT dumped_result EQUAL 0 OR NOT dumped_error STREQUAL "")
  message(FATAL_ERROR "Dumped translation or intermediate verification failed: ${dumped_error}")
endif()
file(GLOB passes "${OUTPUT_DIR}/passes/*.bc")
list(LENGTH passes count)
if(count LESS 7)
  message(FATAL_ERROR "Expected IR dumps from translation through final metadata")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files
  "${OUTPUT_DIR}/plain.metallib" "${OUTPUT_DIR}/dumped.metallib" RESULT_VARIABLE different)
if(NOT different EQUAL 0)
  message(FATAL_ERROR "Intermediate dumping changed final shader bytes")
endif()
