module T_FF_pos_rst_n_by_JKFF(
    input T, clk, rst_n,
    output Q, Qn
);

    JK_FF_Pos u1(rst_n ? T : 1'b0, rst_n ? T : 1'b1, clk, Q, Qn);

endmodule