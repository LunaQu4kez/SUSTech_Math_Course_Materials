module lab_a1_p1(
    input A,
    input B,
    input C,
    input D,
    output X,
    output Y
    );

    wire not_a, w1, w2, w3, w4;
    not unot(not_a, A);
    and uand1(w1, B, C);
    or uor1(w2, not_a, w1);
    xnor uxnor(w3, D, not_a);
    nand unand(w4, not_a, w3);
    or uor2(X, w2, w3);
    and uand2(Y, w2, w4);

endmodule