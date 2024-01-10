module lab_a2_p7 (
    input [3:0] in,
    output [0:0] more1
);

parameter data_in = 16'b1110_1000_1000_0000;
MUX_16_1 m0(in, data_in, more1);

endmodule