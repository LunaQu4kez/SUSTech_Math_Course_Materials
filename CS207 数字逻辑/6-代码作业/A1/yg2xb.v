module yg2xb(
    input [3:0] yg,
    output [3:0] xb
    );

    assign xb[3] = yg[3];
    assign xb[2] = yg[2] ^ xb[3];
    assign xb[1] = yg[1] ^ xb[2];
    assign xb[0] = yg[0] ^ xb[1];

endmodule