import org.junit.jupiter.api.Test;


import static org.junit.jupiter.api.Assertions.*;


public class A6LocalTest {
    private static String gridString(char[][] grids) {
        var sb = new StringBuilder();
        for (char[] line : grids) {
            sb.append(line);
            sb.append("\n");
        }
        return sb.toString();
    }

    @Test
    void test1() {
        var canvas = new AvoidConflictShapeCanvas(20, 20);
        assertTrue(canvas.addShape(2, 4, 'A', 2));
        assertFalse(canvas.addShape(1, 1, 'Z', 2));
        assertFalse(canvas.addShape(18, 18, 'Z', 3, 5, 3));
        assertFalse(canvas.addShape(18, 18, 'Z', 8));
        assertTrue(canvas.addShape(7, 7, 'B', 3, 5, 1));
        assertFalse(canvas.addShape(1, 3, 'Z', 2, 3, 3));
        assertFalse(canvas.addShape(18, 18, 'Z', 2, 3, 3));
        assertFalse(canvas.addShape(19, 19, 'Z', 2));
        assertFalse(canvas.addShape(6, 6, 'Z', 19, 18, 3));
    }
    @Test
    void test2() {
        var canvas = new AvoidConflictShapeCanvas(10, 10);
        assertTrue(canvas.addShape(0, 0, 'A', 1));
        assertTrue(canvas.addShape(5, 5, 'B', 2, 1, 2));
        assertFalse(canvas.addShape(5, 5, 'C', 2, 1, 2));
        assertEquals(2, canvas.getShapeCount());
    }
    @Test
    void test3() {
        var canvas = new AvoidConflictShapeCanvas(15, 20);
        assertTrue(canvas.addShape(1, 2, 'A', 3));
        assertFalse(canvas.addShape(1, 1, 'B', 12, 5, 1));
        assertFalse(canvas.addShape(12, 15, 'C', 3, 5, 2));
        assertTrue(canvas.addShape(11, 17, 'Z', 3, 2, 3));
        assertFalse(canvas.addShape(6, 3, 'D', 7, 17, 3));
        assertEquals("""
                                   \s
                  AAAAAA           \s
                  AAAAAA           \s
                  AAAAAA           \s
                  AAAAAA           \s
                  AAAAAA           \s
                  AAAAAA           \s
                                   \s
                                   \s
                                   \s
                                   \s
                                  ZZ
                                 ZZZ
                                   \s
                                   \s
                """, gridString(canvas.getCanvas()));
    }
    @Test
    void test4() {
        var canvas = new AvoidConflictShapeCanvas(20, 20);

        assertTrue(canvas.addShape(0, 0, 'A', 1));
        assertFalse(canvas.addShape(1, 1, 'Z', 1));
        assertTrue(canvas.addShape(2, 0, 'B', 3, 5, 1));
        assertFalse(canvas.addShape(0, 0, 'Z', 3, 5, 1));
        assertTrue(canvas.addShape(12, 15, 'C', 3, 5, 2));
        assertTrue(canvas.addShape(6, 3, 'D', 7, 12, 3));
        assertFalse(canvas.addShape(6, 6, 'Z', 7, 12, 3));

        assertEquals(4, canvas.getShapeCount());
        assertEquals("""
                AA                 \s
                AA                 \s
                B                  \s
                BB                 \s
                BB                 \s
                BBB                \s
                BBB      D         \s
                        DD         \s
                        DD         \s
                       DDD         \s
                       DDD         \s
                      DDDD         \s
                     DDDDD     CCC \s
                     DDDDD     CCC \s
                    DDDDDD      CC \s
                    DDDDDD      CC \s
                   DDDDDDD       C \s
                   DDDDDDD         \s
                                   \s
                                   \s
                """, gridString(canvas.getCanvas()));
    }
    @Test
    void test5() {
        var canvas = new AvoidConflictShapeCanvas(20, 20);

        assertTrue(canvas.addShape(1, 1, 'M', 1));
        assertFalse(canvas.addShape(0, 0, 'A', 1));
        assertTrue(canvas.addShape(6, 6, 'N', 7, 12, 3));
        assertFalse(canvas.addShape(6, 6, 'Z', 7, 12, 3));
        assertTrue(canvas.addShape(2, 0, 'B', 3, 5, 1));
        assertFalse(canvas.addShape(0, 0, 'Z', 3, 5, 1));
        assertTrue(canvas.addShape(12, 15, 'C', 3, 5, 2));
        assertFalse(canvas.addShape(6, 3, 'D', 7, 12, 3));
        assertEquals(4, canvas.getShapeCount());
        assertEquals("""
                                   \s
                 MM                \s
                BMM                \s
                BB                 \s
                BB                 \s
                BBB                \s
                BBB         N      \s
                           NN      \s
                           NN      \s
                          NNN      \s
                          NNN      \s
                         NNNN      \s
                        NNNNN  CCC \s
                        NNNNN  CCC \s
                       NNNNNN   CC \s
                       NNNNNN   CC \s
                      NNNNNNN    C \s
                      NNNNNNN      \s
                                   \s
                                   \s
                """, gridString(canvas.getCanvas()));
        assertEquals("[Circle: (1,1) area=4 pattern=M, RightTriangle: (2,0) area=11 pattern=B, RightTriangle: (12,15) area=11 pattern=C, RightTriangle: (6,6) area=51 pattern=N]", canvas.getShapesByArea().toString());
        assertEquals("[Circle: (1,1) area=4 pattern=M, RightTriangle: (2,0) area=11 pattern=B, RightTriangle: (6,6) area=51 pattern=N, RightTriangle: (12,15) area=11 pattern=C]", canvas.getShapesByLocation().toString());
    }

    @Test
    void test6() {
        var canvas = new OverLapShapeCanvas(16, 12);
        assertTrue(canvas.addShape(0, 0, 'A', 1));
        assertTrue(canvas.addShape(2, 2, 'B', 3, 5, 1));
        assertTrue(canvas.addShape(6, 6, 'C', 3, 5, 2));
        assertTrue(canvas.addShape(6, 3, 'D', 7, 12, 3));
        assertFalse(canvas.addShape(6, 6, 'E', 19, 20, 3));
    }
    @Test
    void test7() {
        var canvas = new OverLapShapeCanvas(12, 16);
        assertTrue(canvas.addShape(6, 4, 'A', 1));
        assertFalse(canvas.addShape(10, 15, 'B', 18));
        assertFalse(canvas.addShape(15, 19, 'C', 4));
        assertFalse(canvas.addShape(0, 18, 'D', 12, 12, 2));
        assertTrue(canvas.addShape(2, 2, 'X', 3, 5, 1));
        assertTrue(canvas.addShape(2, 2, 'Y', 3, 5, 0));

        assertEquals(3, canvas.getShapeCount());
    }

    @Test
    void test8() {
        var canvas = new OverLapShapeCanvas(12, 16);
        assertTrue(canvas.addShape(6, 4, 'A', 1));
        assertFalse(canvas.addShape(10, 15, 'B', 18));
        assertFalse(canvas.addShape(15, 19, 'C', 4));
        assertFalse(canvas.addShape(0, 18, 'D', 12, 12, 2));
        assertTrue(canvas.addShape(2, 2, 'E', 3, 5, 1));
        assertTrue(canvas.addShape(2, 2, 'F', 3, 5, 0));
        assertTrue(canvas.addShape(6, 6, 'G', 7, 12, 3));
        assertTrue(canvas.addShape(6, 6, 'H', 3, 5, 0));
        assertEquals("""
                               \s
                               \s
                  FFF          \s
                  FFF          \s
                  FF           \s
                  FFE          \s
                  FEEAHHH   G  \s
                    AAHHH  GG  \s
                      HH   GG  \s
                      HH  GGG  \s
                      H   GGG  \s
                         GGGG  \s
                """, gridString(canvas.getCanvas()));
    }
    @Test
    void test9() {
        var canvas = new OverLapShapeCanvas(20, 20);
        assertTrue(canvas.addShape(6, 4, 'A', 1));
        assertFalse(canvas.addShape(19, 19, 'B', 19));
        assertTrue(canvas.addShape(15, 19, 'C', 4));
        assertTrue(canvas.addShape(0, 18, 'D', 12, 12, 2));
        assertTrue(canvas.addShape(2, 2, 'E', 3, 5, 1));
        assertTrue(canvas.addShape(2, 2, 'F', 3, 5, 0));
        assertTrue(canvas.addShape(6, 6, 'G', 7, 12, 3));
        assertTrue(canvas.addShape(6, 6, 'H', 3, 5, 0));
        assertTrue(canvas.addShape(6, 6, 'I', 5));
        assertFalse(canvas.addShape(18, 19, 'J', 5, 5, 3));

        assertEquals(8, canvas.getShapeCount());
        assertEquals("""
                                  DD
                                   D
                  FFF              \s
                  FFF              \s
                  FF               \s
                  FFE              \s
                  FEEAHHIIIIII     \s
                    AAHIIIIIIII    \s
                      IIIIIIIIII   \s
                      IIIIIIIIII   \s
                      IIIIIIIIII   \s
                      IIIIIIIIII   \s
                      IIIIIIIIII   \s
                      IIIIIIIIII   \s
                       IIIIIIII    \s
                       GIIIIII     \s
                      GGGGGGG      C
                      GGGGGGG      C
                                   C
                                   C
                """, gridString(canvas.getCanvas()));
    }
    @Test
    void test10() {
        var canvas = new OverLapShapeCanvas(20, 20);
        assertTrue(canvas.addShape(2, 2, 'F', 3, 5, 0));
        assertTrue(canvas.addShape(6, 6, 'G', 7, 12, 3));
        assertTrue(canvas.addShape(15, 19, 'C', 4));
        assertTrue(canvas.addShape(6, 4, 'A', 1));
        assertFalse(canvas.addShape(19, 19, 'B', 18));
        assertTrue(canvas.addShape(6, 6, 'I', 5));

        assertTrue(canvas.addShape(0, 18, 'D', 12, 12, 2));

        assertTrue(canvas.addShape(2, 2, 'E', 3, 5, 1));


        assertTrue(canvas.addShape(6, 6, 'H', 3, 5, 0));

        assertFalse(canvas.addShape(18, 19, 'J', 5, 5, 3));

        assertEquals(8, canvas.getShapeCount());
        assertEquals("""
                                  DD
                                   D
                  EFF              \s
                  EEF              \s
                  EE               \s
                  EEE              \s
                  EEEAHHHIIIII     \s
                    AAHHHIIIIII    \s
                      HHIIIIIIII   \s
                      HHIIIIIIII   \s
                      HIIIIIIIII   \s
                      IIIIIIIIII   \s
                      IIIIIIIIII   \s
                      IIIIIIIIII   \s
                       IIIIIIII    \s
                       GIIIIII     \s
                      GGGGGGG      C
                      GGGGGGG      C
                                   C
                                   C
                """, gridString(canvas.getCanvas()));
        assertEquals("[Circle: (6,4) area=4 pattern=A, RightTriangle: (2,2) area=11 pattern=E, RightTriangle: (2,2) area=11 pattern=F, RightTriangle: (6,6) area=11 pattern=H, RightTriangle: (6,6) area=51 pattern=G, Circle: (15,19) area=60 pattern=C, RightTriangle: (0,18) area=78 pattern=D, Circle: (6,6) area=88 pattern=I]", canvas.getShapesByArea().toString());
        assertEquals("[RightTriangle: (0,18) area=78 pattern=D, RightTriangle: (2,2) area=11 pattern=E, RightTriangle: (2,2) area=11 pattern=F, Circle: (6,4) area=4 pattern=A, RightTriangle: (6,6) area=51 pattern=G, RightTriangle: (6,6) area=11 pattern=H, Circle: (6,6) area=88 pattern=I, Circle: (15,19) area=60 pattern=C]", canvas.getShapesByLocation().toString());
    }
}
