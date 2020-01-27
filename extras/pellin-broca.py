import sympy as sy

# Work out formulas describing a generalized Pellin-Broca prism

# Generalized prism is a 4-sided prism with fixed inner angles, and a particular pivot point.
# We have sides A,B,C,D, and angles a,b,c,d.
#                 C
#               ---------------
#      ----------            c |
#      |  d                    | B
#   D |                        |
#    |  a                    b |
#    ---------------------------
#                   A
# A is the side of the prism where the input beam is incident, B is the exit plane, C is the reflection plane.
# we have a characteristic angle qr (typically 30 deg)
# The angles (a,b,c,d) are fixed as (90-qr,90,45+qr,135).
# We want an expression for C and D in terms of A and B.
# Strategy is to demand that a walk around the perimeter leads back to the starting point.
# The walk starts at vertex AD and proceeds counter-clockwise (see diagram above).

A,B,C,D,a,b,c,d,qr = sy.symbols('A B C D a b c d qr')
def num(obj,A0,B0):
    # Plug in actual numbers A0 and B0
    return sy.N(obj.subs([(A,A0),(B,B0)]))

a = (90-qr)*sy.pi/180
b = 90*sy.pi/180
c = (45+qr)*sy.pi/180
d = 135*sy.pi/180
# The following are angular directions of the walker along a given side
qA = sy.sympify(0)
qB = qA + sy.pi - b
qC = qB + sy.pi - c
qD = qC + sy.pi - d
print('Walking angles:')
print(qA)
print(qB)
print(qC)
print(qD)

# Determine the lengths of the sides
C = -A/sy.cos(qC) - D*sy.cos(qD)/sy.cos(qC)
D1 = sy.simplify(sy.solveset(B+C*sy.sin(qC)+D*sy.sin(qD),D).args[0])
C1 = sy.simplify(C.subs(D,D1))
C = C1
D = D1
print('Given sides A and B, we have:')
print('C =',C)
print('D =',D)
# Determine the coordinates of the vertices
DA = (sy.sympify(0),sy.sympify(0))
AB = (A,sy.sympify(0))
BC = (A,B)
CD = (A + C*sy.cos(qC),B + C*sy.sin(qC))
print('Coordinates of vertices:')
print('DA =',DA)
print('AB =',AB)
print('BC =',BC)
print('CD =',CD)

# Find the pivot point, defined as the intersection of the bisector of AB with C.
x = sy.symbols('x')
x = sy.solveset(A-x - (B-(A-x)*sy.tan(sy.pi/2-c)),x).args[0] # note slope = 1 for first line
pivot = (x,A-x)
print('pivot point =',pivot)
