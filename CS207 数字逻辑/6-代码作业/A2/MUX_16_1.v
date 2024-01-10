module MUX_16_1(
    input [3:0] sel,
    input [15:0] data_in,
    output reg data_out
);

always @ * begin
    case(sel)
        4'b0000: data_out = data_in[0];
        4'b0001: data_out = data_in[1];
        4'b0010: data_out = data_in[2];
        4'b0011: data_out = data_in[3];
        4'b0100: data_out = data_in[4];
        4'b0101: data_out = data_in[5];
        4'b0110: data_out = data_in[6];
        4'b0111: data_out = data_in[7];
        4'b1000: data_out = data_in[8];
        4'b1001: data_out = data_in[9];
        4'b1010: data_out = data_in[10];
        4'b1011: data_out = data_in[11];
        4'b1100: data_out = data_in[12];
        4'b1101: data_out = data_in[13];
        4'b1110: data_out = data_in[14];
        4'b1111: data_out = data_in[15];
    endcase
end

endmodule