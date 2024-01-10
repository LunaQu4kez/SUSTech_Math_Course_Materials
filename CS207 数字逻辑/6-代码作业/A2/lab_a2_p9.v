module lab_a2_p9(
    input [3:0] in,
    output [0:0] more1
);

wire [15:0] out;
Decoder_4_16 d1(in, out);
or o1(more1, out[7], out[11], out[13], out[14], out[15]);

endmodule