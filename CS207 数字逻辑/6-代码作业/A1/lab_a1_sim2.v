module lab_a1_sim2();

reg [3:0] in_tb;
wire [3:0] out_xb2yg, out_yg2xb;
xb2yg u1(.xb(in_tb), .yg(out_xb2yg));
yg2xb u2(.yg(in_tb), .xb(out_yg2xb));

initial begin
    $monitor ("%d %d %d", in_tb, out_xb2yg, out_yg2xb);
    {in_tb} = 4'b0000;
    repeat(15) #10 {in_tb} = {in_tb} + 1;
    #10 $finish;
end

endmodule