program sin_function
  implicit none
  real :: w, x, d, expf, w0, phi, a, result
  integer :: i, n
  real, parameter :: pi = 3.1415926535897932384626433832795
  
  ! 설정값
  n = 1000  ! 구간을 몇 등분할 것인지 결정
  d = 1.5
  w0=15
  ! 계산
  do i = 0, n
    x = 2.0 * pi * i / n
    
    w=(w0**2-d**2)**.5
    phi=atan(-d/w)
    a=1/(2*cos(phi))
    expf=exp(-d*x)

    result = expf*2*a*cos(phi+w*x)
    write(*, '(F8.4, F10.6)') x, result
  end do
  
end program sin_function

