module MemUnit16_8 (
    input clk, rw, rst_n, // A rw of 1 indicates a read operation, and a rw of 0 indicates a write operation.
    input [3:0] addr, // address to be read or written
    input [7:0] data_in, // when in the write state, assign the value to the corresponding address
    output reg [7:0] data_out, // Data read in read state
    output reg data_valid //represents the data_out is valid
);

    reg [127:0] mem;

    always @(posedge clk or negedge rst_n) begin
        if (rst_n == 0) begin
            mem <= 128'h00000000;
        end
        else if (rw == 0) begin
            mem[addr*8+7] <= data_in[7];
            mem[addr*8+6] <= data_in[6];
            mem[addr*8+5] <= data_in[5];
            mem[addr*8+4] <= data_in[4];
            mem[addr*8+3] <= data_in[3];
            mem[addr*8+2] <= data_in[2];
            mem[addr*8+1] <= data_in[1];
            mem[addr*8+0] <= data_in[0];
        end
    end

    always @(rw, rst_n, addr) begin
        data_out[7] = (rw == 1 & rst_n == 1) ? mem[addr*8+7] : 0;
        data_out[6] = (rw == 1 & rst_n == 1) ? mem[addr*8+6] : 0;
        data_out[5] = (rw == 1 & rst_n == 1) ? mem[addr*8+5] : 0;
        data_out[4] = (rw == 1 & rst_n == 1) ? mem[addr*8+4] : 0;
        data_out[3] = (rw == 1 & rst_n == 1) ? mem[addr*8+3] : 0;
        data_out[2] = (rw == 1 & rst_n == 1) ? mem[addr*8+2] : 0;
        data_out[1] = (rw == 1 & rst_n == 1) ? mem[addr*8+1] : 0;
        data_out[0] = (rw == 1 & rst_n == 1) ? mem[addr*8+0] : 0;
        data_valid = rw & rst_n == 1;
    end



endmodule