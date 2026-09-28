"""Item 6: verify the hand-rolled OLS against an INDEPENDENT algorithm
(Householder QR least-squares), which does not go through normal equations."""
import random, math

def handrolled(X, y, ridge=1e-6):
    """EXACT copy of format_confound.py's solver."""
    k = len(X[0])
    XtX = [[sum(r[i]*r[j] for r in X) + (ridge if i==j else 0.0) for j in range(k)] for i in range(k)]
    Xty = [sum(r[i]*yy for r, yy in zip(X, y)) for i in range(k)]
    A = [row[:] + [b] for row, b in zip(XtX, Xty)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        for r in range(k):
            if r != col and A[col][col]:
                f = A[r][col]/A[col][col]
                for c in range(col, k+1):
                    A[r][c] -= f*A[col][c]
    return [A[i][k]/A[i][i] for i in range(k)]

def qr_lstsq(X, y):
    """Householder QR least squares -- independent of normal equations."""
    n, k = len(X), len(X[0])
    A = [row[:] for row in X]; b = y[:]
    for j in range(k):
        nrm = math.sqrt(sum(A[i][j]**2 for i in range(j, n)))
        if nrm == 0: continue
        if A[j][j] > 0: nrm = -nrm
        v = [0.0]*n
        for i in range(j, n): v[i] = A[i][j]
        v[j] -= nrm
        vn = math.sqrt(sum(v[i]**2 for i in range(j, n)))
        if vn < 1e-300: continue
        for i in range(j, n): v[i] /= vn
        for c in range(j, k):
            s = 2*sum(v[i]*A[i][c] for i in range(j, n))
            for i in range(j, n): A[i][c] -= s*v[i]
        s = 2*sum(v[i]*b[i] for i in range(j, n))
        for i in range(j, n): b[i] -= s*v[i]
    beta = [0.0]*k
    for i in range(k-1, -1, -1):
        if abs(A[i][i]) < 1e-12: beta[i] = 0.0; continue
        beta[i] = (b[i] - sum(A[i][c]*beta[c] for c in range(i+1, k)))/A[i][i]
    return beta

def r2_of(X, y, beta):
    pred=[sum(bb*v for bb,v in zip(beta,r)) for r in X]
    resid=[yy-p for yy,p in zip(y,pred)]
    ybar=sum(y)/len(y)
    return 1 - sum(e*e for e in resid)/sum((yy-ybar)**2 for yy in y)

rng = random.Random(1)
print("=== random well-conditioned designs ===")
for t in range(4):
    n,k=300,8
    X=[[rng.gauss(0,1) for _ in range(k)] for _ in range(n)]
    bt=[rng.gauss(0,1) for _ in range(k)]
    y=[sum(b*v for b,v in zip(bt,r))+rng.gauss(0,0.3) for r in X]
    a,b=handrolled(X,y),qr_lstsq(X,y)
    print(f"  trial {t}: max|beta diff|={max(abs(u-v) for u,v in zip(a,b)):.3e}"
          f"  R2 handrolled={r2_of(X,y,a):.6f} qr={r2_of(X,y,b):.6f}")

print("\n=== ACTUAL format_confound design: 8 std feats + 15 lang dummies, NO intercept ===")
n,nf,nl=1200,8,15
X=[];y=[]
for i in range(n):
    f=[rng.gauss(0,1) for _ in range(nf)]
    d=[0.0]*nl; d[i%nl]=1.0
    X.append(f+d); y.append(sum(f[:3])+ (i%nl)*0.1 + rng.gauss(0,0.2))
a,b=handrolled(X,y),qr_lstsq(X,y)
print(f"  max|beta diff|={max(abs(u-v) for u,v in zip(a,b)):.3e}")
print(f"  R2 handrolled={r2_of(X,y,a):.6f}  qr={r2_of(X,y,b):.6f}")

print("\n=== ablate 'format-only': 8 feats + explicit intercept ===")
X2=[[rng.gauss(0,1) for _ in range(nf)]+[1.0] for _ in range(n)]
y2=[rng.gauss(0,1) for _ in range(n)]
a,b=handrolled(X2,y2),qr_lstsq(X2,y2)
print(f"  max|beta diff|={max(abs(u-v) for u,v in zip(a,b)):.3e}  R2 hr={r2_of(X2,y2,a):.6f} qr={r2_of(X2,y2,b):.6f}")

print("\n=== ablate 'language-only': 15 dummies, no intercept ===")
X3=[]
for i in range(n):
    d=[0.0]*nl; d[i%nl]=1.0
    X3.append(d)
y3=[(i%nl)*0.1+rng.gauss(0,0.5) for i in range(n)]
a,b=handrolled(X3,y3),qr_lstsq(X3,y3)
print(f"  max|beta diff|={max(abs(u-v) for u,v in zip(a,b)):.3e}  R2 hr={r2_of(X3,y3,a):.6f} qr={r2_of(X3,y3,b):.6f}")
