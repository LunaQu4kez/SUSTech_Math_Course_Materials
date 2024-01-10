module lab_a2_p6 (
    input [3:0] in,
    output reg [0:0] more1
);

always @ * begin
    case(in)
        4'b1111, 4'b0111, 4'b1101, 4'b1011, 4'b1110: more1 = 1;
        default: more1 = 0;
    endcase
end    


endmodule