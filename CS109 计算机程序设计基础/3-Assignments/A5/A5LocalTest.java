import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class A5LocalTest {
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
        var location = new Location(0, 0);
        var circle = new Circle(location, 'C', 1);
        assertEquals(4, circle.area());
        assertEquals("""
                CC
                CC
                """, gridString(circle.getGrids()));
        assertEquals("Circle: (0,0) area=4 pattern=C", circle.toString());
    }

    @Test
    void test2() {
        var location = new Location(1, 1);
        var circle = new Circle(location, '2', 3);
        assertEquals(36, circle.area());
        assertEquals("""
                222222
                222222
                222222
                222222
                222222
                222222
                """, gridString(circle.getGrids()));
        assertEquals("Circle: (1,1) area=36 pattern=2", circle.toString());
    }

    @Test
    void test3() {
        var location = new Location(3, 4);
        var circle = new Circle(location, 'X', 5);
        assertEquals(88, circle.area());
        circle.enlarge();
        assertEquals("""
                  XXXXXXXX \s
                 XXXXXXXXXX\s
                XXXXXXXXXXXX
                XXXXXXXXXXXX
                XXXXXXXXXXXX
                XXXXXXXXXXXX
                XXXXXXXXXXXX
                XXXXXXXXXXXX
                XXXXXXXXXXXX
                XXXXXXXXXXXX
                 XXXXXXXXXX\s
                  XXXXXXXX \s
                """, gridString(circle.getGrids()));
        assertEquals("Circle: (3,4) area=132 pattern=X", circle.toString());
    }

    @Test
    void test4() {
        var location = new Location(9, 9);
        var circle = new Circle(location, '-', 8);
        assertEquals(224, circle.area());
        circle.shrink();
        assertEquals("""
                   --------  \s
                  ---------- \s
                 ------------\s
                --------------
                --------------
                --------------
                --------------
                --------------
                --------------
                --------------
                --------------
                 ------------\s
                  ---------- \s
                   --------  \s
                """, gridString(circle.getGrids()));
        assertEquals("Circle: (9,9) area=172 pattern=-", circle.toString());
    }

    @Test
    void test5() {
        var location = new Location(-100, 100);
        var circle = new Circle(location, '=', 9);
        circle.shrink();
        circle.enlarge();
        circle.shrink();
        circle.shrink();
        circle.enlarge();
        circle.enlarge();
        assertEquals("""
                    ==========   \s
                   ============  \s
                  ============== \s
                 ================\s
                ==================
                ==================
                ==================
                ==================
                ==================
                ==================
                ==================
                ==================
                ==================
                ==================
                 ================\s
                  ============== \s
                   ============  \s
                    ==========   \s
                """, gridString(circle.getGrids()));
        assertEquals("Circle: (-100,100) area=284 pattern==", circle.toString());
    }

    @Test
    void test6() {
        var location = new Location(123, 233);
        var triangle = new RightTriangle(location, 'T', 3, 4, Direction.LEFT_DOWN);
        assertEquals(9, triangle.area());
        assertEquals("""
                T \s
                TT\s
                TTT
                TTT
                """, gridString(triangle.getGrids()));
        assertEquals("RightTriangle: (123,233) area=9 pattern=T", triangle.toString());
    }

    @Test
    void test7() {
        var location = new Location(-123456, 99999);
        var triangle = new RightTriangle(location, '@', 6, 6, Direction.RIGHT_UP);
        assertEquals(21, triangle.area());
        assertEquals("""
                @@@@@@
                 @@@@@
                  @@@@
                   @@@
                    @@
                     @
                """, gridString(triangle.getGrids()));
        assertEquals("RightTriangle: (-123456,99999) area=21 pattern=@", triangle.toString());
    }

    @Test
    void test8() {
        var location = new Location(104, 222);
        var triangle = new RightTriangle(location, '~', 2, 2, Direction.RIGHT_DOWN);
        assertEquals(3, triangle.area());
        triangle.enlarge();
        assertEquals("""
                  ~
                 ~~
                ~~~
                """, gridString(triangle.getGrids()));
        assertEquals("RightTriangle: (104,222) area=6 pattern=~", triangle.toString());
    }

    @Test
    void test9() {
        var location = new Location(99, 999);
        var triangle = new RightTriangle(location, '+', 7, 4, Direction.LEFT_UP);
        assertEquals(19, triangle.area());
        triangle.shrink();
        assertEquals("""
                ++++++
                ++++ \s
                ++   \s
                """, gridString(triangle.getGrids()));
        assertEquals("RightTriangle: (99,999) area=12 pattern=+", triangle.toString());
    }

    @Test
    void test10() {
        var location = new Location(10, 10);
        var triangle = new RightTriangle(location, '"', 4, 5, Direction.RIGHT_DOWN);
        assertEquals(14, triangle.area());
        triangle.enlarge();
        triangle.enlarge();
        triangle.shrink();
        triangle.shrink();
        triangle.shrink();
        triangle.enlarge();
        triangle.shrink();
        assertEquals("""
                  "
                 ""
                ""\"
                ""\"
                """, gridString(triangle.getGrids()));
        assertEquals("RightTriangle: (10,10) area=9 pattern=\"", triangle.toString());
    }
}
