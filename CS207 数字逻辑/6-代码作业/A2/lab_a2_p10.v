module lab_a2_p10 ();

reg [3:0] in;
wire [0:0] a2_p5_somin, a2_p5_pomax, a2_p6_more1, a2_p7_more1, a2_p8_more1, a2_p9_more1;
lab_a2_p5 u5(in[3], in[2], in[1], in[0], a2_p5_somin, a2_p5_pomax);
lab_a2_p6 u6(in, a2_p6_more1);
lab_a2_p7 u7(in, a2_p7_more1);
lab_a2_p8 u8(in, a2_p8_more1);
lab_a2_p9 u9(in, a2_p9_more1);

initial begin
    $monitor ("%d %d %d %d %d %d %d", in, a2_p5_somin, a2_p5_pomax, a2_p6_more1, a2_p7_more1, a2_p8_more1, a2_p9_more1);
    in = 4'b0000;
    repeat(15) #10 in = in + 1;
    #10 $finish;
end

endmodule