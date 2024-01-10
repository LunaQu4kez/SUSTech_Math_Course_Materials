module lab_a1_sim1();

reg a_tb, b_tb, c_tb, d_tb;
wire x_tb1, y_tb1, x_tb2, y_tb2;
lab_a1_p1 u1(.A(a_tb), .B(b_tb), .C(c_tb), .D(d_tb), .X(x_tb1), .Y(y_tb1));
lab_a1_p2 u2(.A(a_tb), .B(b_tb), .C(c_tb), .D(d_tb), .X(x_tb2), .Y(y_tb2));

initial begin
    $monitor ("%d %d %d %d %d %d %d %d", a_tb, b_tb, c_tb, d_tb, x_tb1, y_tb1, x_tb2, y_tb2);
    {a_tb, b_tb, c_tb, d_tb} = 4'b0000;
    repeat(15) #10 {a_tb, b_tb, c_tb, d_tb} = {a_tb, b_tb, c_tb, d_tb} + 1;
    #10 $finish;
end

endmodule