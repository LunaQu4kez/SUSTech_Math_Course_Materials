module moor_s1s2_rst_asyn(
    input clk, rst, x,
    output reg [1:0]state, next_s
);
    
    always @(state, x) begin
        if (x == 1) begin
            next_s = state;
        end
        else begin
            next_s = ~state;
        end
    end

    always @(posedge clk, posedge rst) begin
        if (rst) state <= 2'b01;
        else state <= next_s;
    end
    
endmodule