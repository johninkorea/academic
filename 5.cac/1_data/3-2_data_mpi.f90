program mpi_sin_function
  use mpi
  
  implicit none
  integer :: ierr, myrank, nprocs, i, n
  real :: x, result, sendbuf(1), recvbuf(1)
  real, parameter :: pi = 3.1415926535897932384626433832795
  
  ! 설정값
  n = 1000  ! 구간을 몇 등분할 것인지 결정
  
  call MPI_INIT(ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  
  ! 계산
  do i = myrank, n, nprocs
    x = 2.0 * pi * i / n
    result = sin(x)
    write(*, '(I4, F8.4, F10.6)') myrank, x, result
    
    ! 결과를 모으기 위해 모든 프로세스에서 결과값을 전달
    sendbuf(1) = result
    call MPI_ALLGATHER(sendbuf, 1, MPI_REAL, recvbuf, 1, MPI_REAL, MPI_COMM_WORLD, ierr)
    
    ! 결과값을 모든 프로세스에서 출력
    if (myrank == 0) then
      write(*, '(F8.4, F10.6)') x, recvbuf(1)
    end if
  end do
  
  call MPI_FINALIZE(ierr)
  
end program mpi_sin_function

