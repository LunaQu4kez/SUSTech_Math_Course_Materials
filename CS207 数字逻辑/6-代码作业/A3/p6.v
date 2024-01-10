module check_dif (
    input clk, rst, x,
    output reg z
);

    reg last_two_x, last_one_x, temp_x;

    always @(posedge clk) begin
        if (rst) begin
            z <= 0;
        end
        else begin
            if (last_two_x == last_one_x) begin
                z <= 0;
            end
            else begin
                z <= 1;
            end
        end
        temp_x <= x;
    end

    always @(negedge clk) begin
        if (rst) begin
            last_one_x <= 0;
            last_two_x <= 0;
        end
        else begin
            last_one_x <= temp_x;
            last_two_x <= last_one_x;
        end
    end
    
endmodule