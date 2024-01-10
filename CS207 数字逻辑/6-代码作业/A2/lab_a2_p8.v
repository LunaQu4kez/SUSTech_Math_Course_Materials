module lab_a2_p8 (
    input [3:0] in,
    output [0:0] more1
);

wire [7:0] x = in[3] ? 8'b1110_1000 : 8'b1000_0000;
MUX_8_1 m1(in[2:0], x, more1);

endmodule