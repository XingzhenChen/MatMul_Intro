open_project hls_top
set_top matrix_multiply
add_files src/mm.cpp
add_files -tb testbench/test.cpp
open_solution "solution1" -flow_target vivado
set_part {xcvc1902-vsva2197-2MP-e-S}
create_clock -period 3.3 -name default
config_export -format ip_catalog -rtl verilog
csim_design
#csynth_design
#cosim_design
#cosim_design -tool xsim -rtl verilog -coverage -trace_level all -wave_debug
# export_design -rtl verilog -format ip_catalog
