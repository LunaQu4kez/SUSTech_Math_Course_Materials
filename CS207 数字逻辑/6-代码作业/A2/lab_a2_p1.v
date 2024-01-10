module lab_a2_p1 (
    input [2:0] a, b,
    output reg sum_flag,
    output reg [2:0] sum_absolut
);

    always @(*) begin
        case ({a, b})
            6'b100_100: begin
                sum_flag = 1'b1;
                sum_absolut = 3'b000;
            end
            6'b100_101, 6'b101_100: begin
                sum_flag = 1'b1;
                sum_absolut = 3'b111;
            end
            6'b101_101, 6'b110_100, 6'b100_110: begin
                sum_flag = 1'b1;
                sum_absolut = 3'b110;
            end
            6'b100_111, 6'b110_101, 6'b101_110, 6'b111_100: begin
                sum_flag = 1'b1;
                sum_absolut = 3'b101;
            end
            6'b100_000, 6'b101_111, 6'b110_110, 6'b111_101, 6'b000_100: begin
                sum_flag = 1'b1;
                sum_absolut = 3'b100;
            end
            6'b100_001, 6'b101_000, 6'b110_111, 6'b111_110, 6'b000_101, 6'b001_100: begin
                sum_flag = 1'b1;
                sum_absolut = 3'b011;
            end
            6'b100_010, 6'b101_001, 6'b110_000, 6'b111_111, 6'b000_110, 6'b001_101, 6'b010_100: begin
                sum_flag = 1'b1;
                sum_absolut = 3'b010;
            end
            6'b100_011, 6'b101_010, 6'b110_001, 6'b111_000, 6'b000_111, 6'b001_110, 6'b010_101, 6'b011_100: begin
                sum_flag = 1'b1;
                sum_absolut = 3'b001;
            end
            6'b101_011, 6'b110_010, 6'b111_001, 6'b000_000, 6'b001_111, 6'b010_110, 6'b011_101: begin
                sum_flag = 1'b0;
                sum_absolut = 3'b000;
            end
            6'b110_011, 6'b111_010, 6'b000_001, 6'b001_000, 6'b010_111, 6'b011_110: begin
                sum_flag = 1'b0;
                sum_absolut = 3'b001;
            end
            6'b111_011, 6'b000_010, 6'b001_001, 6'b010_000, 6'b011_111: begin
                sum_flag = 1'b0;
                sum_absolut = 3'b010;
            end
            6'b000_011, 6'b001_010, 6'b010_001, 6'b011_000: begin
                sum_flag = 1'b0;
                sum_absolut = 3'b011;
            end
            6'b001_011, 6'b010_010, 6'b011_001: begin
                sum_flag = 1'b0;
                sum_absolut = 3'b100;
            end
            6'b010_011, 6'b011_010: begin
                sum_flag = 1'b0;
                sum_absolut = 3'b101;
            end
            6'b011_011: begin
                sum_flag = 1'b0;
                sum_absolut = 3'b110;
            end
        endcase
    end

endmodule