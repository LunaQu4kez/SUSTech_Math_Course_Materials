module clock_div_7 (
    input [0:0] clk, rst_n,
    output reg [0:0] clk_out
);
    
    parameter a0 = 8'h10, a1 = 8'h11, a2 = 8'h12, a3 = 8'h13,
              a4 = 8'h14, a5 = 8'h15, a6 = 8'h16,
              b0 = 8'h20, b1 = 8'h21, b2 = 8'h22, b3 = 8'h23,
              b4 = 8'h24, b5 = 8'h25, b6 = 8'h26;
    
    reg [7:0] state1, next_state1, state2, next_state2;

    always @(state1, rst_n) begin
        if (~rst_n) begin
            next_state1 = a0;
        end
        else begin
            case (state1)
                a0: next_state1 = a1;
                a1: next_state1 = a2;
                a2: next_state1 = a3;
                a3: next_state1 = a4;
                a4: next_state1 = a5;
                a5: next_state1 = a6;
                a6: next_state1 = a0;
                default: next_state1 = a0;
            endcase
        end
    end

    always @(state2, rst_n) begin
        if (~rst_n) begin
            next_state2 = b0;
        end
        else begin
            case (state2)
                b0: next_state2 = b1;
                b1: next_state2 = b2;
                b2: next_state2 = b3;
                b3: next_state2 = b4;
                b4: next_state2 = b5;
                b5: next_state2 = b6;
                b6: next_state2 = b0;
                default: next_state2 = b0;
            endcase
        end
    end

    always @(posedge clk) begin
        state1 <= next_state1;
    end

    always @(negedge clk) begin
        state2 <= next_state2;
    end

    always @(state1, state2, rst_n) begin
        if (rst_n == 0) clk_out = 0;
        else
            if (state1 == a1 & clk_out == 0) begin
                clk_out = 1;
            end
            else if (state2 == b4 & clk_out == 1) begin
                clk_out = 0;
            end
    end

endmodule