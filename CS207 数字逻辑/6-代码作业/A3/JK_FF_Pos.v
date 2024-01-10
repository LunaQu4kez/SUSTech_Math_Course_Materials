module JK_FF_Pos(
    input J,K, clk,
    output reg Q,
    output Qn
);

    assign Qn = ~Q;
    always@(posedge clk) begin
        case({J,K})
            2'b00: Q<=Q;
            2'b01: Q<=1'b0;
            2'b10: Q<=1'b1;
            2'b11: Q<=~Q;
        endcase
    end

endmodule