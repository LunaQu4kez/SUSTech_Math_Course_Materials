module tb_clk_div7();

    reg clk, rst_n;
    wire clk_out;
    clock_div_7 u1(clk, rst_n, clk_out);

    initial begin
        $monitor ("%d %d %d", clk, rst_n, clk_out);
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial fork
        rst_n = 0;
        #16 rst_n = 1;
        #500 $finish;
    join

endmodule