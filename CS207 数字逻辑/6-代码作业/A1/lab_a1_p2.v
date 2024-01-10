module lab_a1_p2(
    input A,
    input B,
    input C,
    input D,
    output X,
    output Y
    );

    assign no_a = ~A;
    assign ua1 = B & C;
    assign uxn = D ~^ no_a;
    assign uo1 = ua1 | no_a;
    assign una = ~(no_a & uxn);
    assign Y = una & uo1;
    assign X = uo1 | uxn;

endmodule