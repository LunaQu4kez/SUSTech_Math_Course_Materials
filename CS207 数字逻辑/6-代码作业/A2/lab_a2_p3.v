module lab_a2_p3(
    input [2:0] a, b,
    output [0:0] tub_sflag_sel,
    output [0:0] tub_sabsolut_sel,
    output [7:0] tub_sflag_control,
    output [7:0] tub_sabsolut_control
);

wire [2:0] abs;
wire [3:0] abs_in;
wire [0:0] flag;
wire [3:0] in;
assign tub_sabsolut_sel = 1'b1;
assign in = (flag ? 4'b1111 : 4'b0000);
assign abs_in[3] = 1'b0;
assign abs_in[2:0] = abs;

lab_a2_p1 u1(.a(a), .b(b), .sum_flag(flag), .sum_absolut(abs));
lab_a2_p3_tub_display f(.in_b4(in), .tub_sel_in(flag), .tub_sel_out(tub_sflag_sel), .tub_control(tub_sflag_control));
lab_a2_p3_tub_display ab(.in_b4(abs_in), .tub_sel_in(1'b1), .tub_control(tub_sabsolut_control));

endmodule