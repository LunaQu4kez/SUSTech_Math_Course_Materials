module tb_moor_s1s2_rst_syn_asyn();

    reg clk, rst, x;
    wire [1:0] state, next_s, state_asy, next_s_asy;
    moor_s1s2_rst_syn dut_sy(clk, rst, x,state, next_s);
    moor_s1s2_rst_asyn dut_asyn(clk, rst, x,state_asy, next_s_asy);

    initial begin
        $monitor ("%d %d %d %d %d %d %d", x, clk, rst, state, next_s, state_asy, next_s_asy);
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial fork
        x = 0;
        forever begin
            #16 x = ~x;
        end
    join

    initial fork
        rst = 0;
        #3 rst = 1;
        #26 rst = 0;
        #120 $finish;
    join

endmodule