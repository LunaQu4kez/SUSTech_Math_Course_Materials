module xb2yg(
    input [3:0] xb,
    output [3:0] yg
    );

    buf ubuf(yg[3], xb[3]);
    xor uxor1(yg[2], xb[3], xb[2]);
    xor uxor2(yg[1], xb[2], xb[1]);
    xor uxor3(yg[0], xb[1], xb[0]);

endmodule