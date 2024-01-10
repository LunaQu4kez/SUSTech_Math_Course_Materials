module tb_T_FF_pos_rst_n();

    reg T, clk, rst_n;
    wire Q, Qn;
    T_FF_pos_rst_n_by_JKFF tff(T, clk, rst_n, Q, Qn);

    initial begin
        $monitor ("%d %d %d %d %d", T, clk, rst_n, Q, Qn);
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
        #100 $finish;
    end

    initial fork
        T = 0;
        forever begin
            #15 T = ~T;
        end
        #100 $finish;
    join

    initial fork
        rst_n = 0;
        #43 rst_n = 1;
    join

endmodule