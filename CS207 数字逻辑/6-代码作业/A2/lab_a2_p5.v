module lab_a2_p5 (
    input [0:0] a, b, c, d,
    output [0:0] more1_somin, more1_pomax
);
    
assign more1_somin = a&b&c&d | a&b&~c&d | a&b&c&~d | ~a&b&c&d | a&~b&c&d;
assign more1_pomax = (a|b|c|d) & (a|b|c|~d) & (a|b|~c|d) & (a|b|~c|~d) & (a|~b|c|d) & (a|~b|c|~d) & 
                    (a|~b|~c|d) & (~a|b|c|d) & (~a|b|c|~d) & (~a|b|~c|d) & (~a|~b|c|d);

endmodule