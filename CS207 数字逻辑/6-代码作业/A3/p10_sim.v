module tb_MemUnit16_8 ();

    reg clk, rw, rst_n;
    reg [3:0] addr;
    reg [7:0] data_in;
    wire [7:0] data_out;
    wire data_valid;
    MemUnit16_8 u1(clk, rw, rst_n, addr, data_in, data_out, data_valid);

    initial begin
        clk = 1;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial fork
        rst_n = 0;
        #1 rst_n = 1;
        #495 $finish;
    join

    initial fork
        rw = 0;
        #240 rw = 1;
    join

    initial fork
        addr = 0;
        forever begin
            #15 addr = addr + 1;
        end
    join

    initial fork
        data_in = 0;
        forever begin
            #15 data_in = data_in + 1;
        end
    join
    
endmodule