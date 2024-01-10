module lab_a2_p4();

reg [2:0] a, b;
wire [0:0] tub_sflag_sel, tub_sabsolut_sel;
wire [7:0] tub_sflag_control, tub_sabsolut_control;
lab_a2_p3 u1(a, b, tub_sflag_sel, tub_sabsolut_sel, tub_sflag_control, tub_sabsolut_control);

initial begin
    $monitor ("%d %d %d %d %d %d", a, b, tub_sflag_sel, tub_sabsolut_sel, tub_sflag_control, tub_sabsolut_control);
    {a, b} = 6'b000_000;
    repeat(63) #10 {a,b} = {a,b} + 1;
    #10 $finish;
end


endmodule