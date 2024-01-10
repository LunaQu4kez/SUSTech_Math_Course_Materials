module lab_a2_p2();

reg [2:0] a, b;
wire sum_flag;
wire [2:0] sum_absolut;
lab_a2_p1 u1(a, b, sum_flag, sum_absolut);

initial begin
    $monitor ("%d %d %d %d", a, b, sum_flag, sum_absolut);
    {a, b} = 6'b000_000;
    repeat(63) #10 {a,b} = {a,b} + 1;
    #10 $finish;
end

endmodule