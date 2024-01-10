module tb_check_dif ();
    
    reg clk, rst, x;
    wire z;
    check_dif u1(clk, rst, x, z);

    initial begin
        $monitor ("%d %d %d %d", clk, rst, x, z);
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial fork
        rst = 1;
        #6 rst = 0;
        #160 $finish;
    join

    initial fork
        x = 0;
        #12 x = 1;
        #48 x = 0;
        #88 x = 1;
        #128 x = 0;
        #148 x = 1;
    join

endmodule