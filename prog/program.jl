using FFTW
using LinearAlgebra

function DCTI(fx, n)
    y = real(FFTW.fft(vcat(fx,fx[n:-1:2])))
    dct = vcat(y[1], y[2:n]+y[2*n:-1:n+2], y[n+1])
    return dct
end

function Clenshaw_Curtis(f, n)
    fx = f.(cos.(pi*(0:n)/n))/2n
    a = DCTI(fx, n)
    w = zeros(length(a))
    w[1:2:end] = 2 ./ (1 .- (0:2:n) .^ 2 )
    return sum(w .* a)
end

function rescale(f, a, b)
    function r(x)
        d = b-a
        return d/2*f(d/2*x + (a+b)/2)
    end
    return r
end

CCintegrate(f, int, N) = Clenshaw_Curtis(rescale(f, int[1], int[2]), N)
