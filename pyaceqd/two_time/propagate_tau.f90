! f2py -c --f90flags="-fopenmp" -m propagate_tau_module propagate_tau.f90 -lopenblas -lgomp

subroutine propagate_tau(dm_tl, rho_init, n_tau, dim, j_start, rho_out)
    implicit none
    integer, intent(in) :: dim, n_tau, j_start
    complex(8), intent(in) :: dm_tl(dim*dim, dim*dim, *)
    complex(8), intent(in) :: rho_init(dim*dim)
    complex(8), intent(out) :: rho_out(dim*dim, n_tau+1)

    integer :: k

    ! Initial condition
    rho_out(:,1) = rho_init

    do k = 1, n_tau
        call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_tl(:,:,j_start+k), dim*dim, &
                   rho_out(:,k), 1, (0.0d0,0.0d0), rho_out(:,k+1), 1)
    end do
end subroutine

! subroutine calc_onetime(dm_tl, rho_init, n_tau, dim, opA, opB, time, time_sparse, result)
!     implicit none
!     integer, intent(in) :: dim, n_tau
!     complex(8), intent(in) :: dm_tl(dim*dim, dim*dim, *)
!     complex(8), intent(in) :: rho_init(dim*dim)
!     complex(8), intent(in) :: opA(dim,dim), opB(dim,dim)
!     real(8), intent(in) :: time(*), time_sparse(*)
!     complex(8), intent(out) :: result(n_tau+1)


! end subroutine calc_onetime

module utils
contains
    function itoa(i) result(str)
        integer, intent(in) :: i
        character(len=20) :: str
        write(str, '(I0)') i
    end function itoa
end module utils


subroutine calc_onetime(dm_tl, rho_init, n_tau, n_t, n_tfull, dim, opA, opB, opC, time, time_sparse, result)
    use utils
    implicit none
    integer, intent(in) :: dim, n_tau, n_t, n_tfull
    complex(8), intent(in) :: dm_tl(dim*dim, dim*dim, n_tfull-1)
    complex(8), intent(in) :: rho_init(dim*dim)
    complex(8), intent(in) :: opA(dim,dim), opB(dim,dim), opC(dim,dim)
    real(8), intent(in) :: time(n_tfull), time_sparse(n_t)
    complex(8), intent(out) :: result(n_t, n_tau+1)

    ! Locals
    integer :: j, i, k, l
    complex(8) :: rho_vec(dim*dim), rho_res(dim*dim)
    complex(8) :: rho_mtx(dim, dim), tmp(dim, dim)
    complex(8) :: rho_temp(dim*dim)
    real(8) :: time_round(n_tfull), time_sparse_round(n_t)



    rho_vec = rho_init

    ! round time and time_sparse to nearest 1e-6
    time_round = time !nint(time * 1.0d6, kind=8) / 1.0d6
    time_sparse_round = time_sparse !nint(time_sparse * 1.0d6, kind=8) / 1.0d6

    ! Step 1: compute result(1) = Tr(opA * opB * opC * rho_t)
    rho_mtx = reshape(rho_vec, [dim, dim])
    tmp = matmul(opC, rho_mtx)
    tmp = matmul(opB, tmp)
    tmp = matmul(opA, tmp)
    result(1,1) = sum([(tmp(l,l), l=1,dim)])
    j = 1
    do i=1, n_t
        ! Step 2: propagate rho_init up to time(j)
        do while (time_round(j) < time_sparse_round(i))
            call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_tl(:,:,j), dim*dim, &
                   rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            rho_vec = rho_res
            j = j + 1
        end do
        ! Step 3: compute result(1) = Tr(opA * opB * rho_t), e.g. tau = 0 value
        rho_mtx = reshape(rho_vec, [dim, dim])
        tmp = matmul(opC, rho_mtx)
        tmp = matmul(opB, tmp)
        tmp = matmul(opA, tmp)
        result(i,1) = sum([(tmp(l,l), l=1,dim)])
        ! write(*,*) "tau = 0, time = ", time_sparse_round(i), " result(1) = ", rho_vec(4)
        ! Step 4: apply opB to rho_t and vectorize again
        tmp = matmul(opC, rho_mtx)
        tmp = matmul(tmp, opA)
        rho_res = reshape(tmp, [dim*dim])

        ! ! Step 5: propagate tau evolution starting from index j+1
        ! call propagate_tau(dm_tl, rho_res, n_tau, dim, j, rho_out)
        ! Step 6: compute result(2:) = Tr(opA * rho_tau)
        do k = 2, n_tau+1
            call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_tl(:,:,j-2+k), dim*dim, &
                   rho_res, 1, (0.0d0,0.0d0), rho_temp, 1)
            tmp = reshape(rho_temp, [dim, dim])
            tmp = matmul(opB, tmp)
            result(i,k) = sum([(tmp(l,l), l=1,dim)])
            rho_res = rho_temp
        end do
        write(*,'(A,I5,A)', advance='no') char(13)//'Progress: ', i, ' / '//trim(adjustl(itoa(n_t)))
    end do
end subroutine calc_onetime

subroutine calc_onetime_parallel(dm_tl, rho_init, n_tau, n_t, n_tfull, dim, opA, opB, opC, time, time_sparse, result)
    use utils
    implicit none
    integer, intent(in) :: dim, n_tau, n_t, n_tfull
    complex(8), intent(in) :: dm_tl(dim*dim, dim*dim, n_tfull-1)
    complex(8), intent(in) :: rho_init(dim*dim)
    complex(8), intent(in) :: opA(dim,dim), opB(dim,dim), opC(dim,dim)
    real(8), intent(in) :: time(n_tfull), time_sparse(n_t)
    complex(8), intent(out) :: result(n_t, n_tau+1)

    ! Locals
    integer :: j, i, k, l
    complex(8) :: rho_vec(dim*dim), rho_res(dim*dim)
    complex(8) :: rho_mtx(dim, dim), tmp(dim, dim)
    complex(8) :: rho_temp(dim*dim)
    real(8) :: time_round(n_tfull), time_sparse_round(n_t)

    complex(8) :: rho_buffer(dim*dim, n_t)
    integer :: j_array(n_t)


    rho_vec = rho_init

    ! round time and time_sparse to nearest 1e-6
    time_round = time !nint(time * 1.0d6, kind=8) / 1.0d6
    time_sparse_round = time_sparse !nint(time_sparse * 1.0d6, kind=8) / 1.0d6

    ! Step 1: compute result(1) = Tr(opA * opB * opC * rho_t)
    rho_mtx = reshape(rho_vec, [dim, dim])
    tmp = matmul(opC, rho_mtx)
    tmp = matmul(opB, tmp)
    tmp = matmul(opA, tmp)
    result(1,1) = sum([(tmp(l,l), l=1,dim)])
    j = 1
    do i=1, n_t
        ! Step 2: propagate rho_init up to time(j)
        do while (time_round(j) < time_sparse_round(i))
            call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_tl(:,:,j), dim*dim, &
                   rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            rho_vec = rho_res
            j = j + 1
        end do
        
        ! Step 3: compute result(1) = Tr(opA * opB * rho_t), e.g. tau = 0 value
        rho_mtx = reshape(rho_vec, [dim, dim])
        tmp = matmul(opC, rho_mtx)
        tmp = matmul(opB, tmp)
        tmp = matmul(opA, tmp)
        result(i,1) = sum([(tmp(l,l), l=1,dim)])
        ! write(*,*) "tau = 0, time = ", time_sparse_round(i), " result(1) = ", rho_vec(4)
        ! Step 4: apply opB to rho_t and vectorize again
        tmp = matmul(opC, rho_mtx)
        tmp = matmul(tmp, opA)
        rho_res = reshape(tmp, [dim*dim])
        rho_buffer(:,i) = rho_res
        j_array(i) = j
        ! ! Step 5: propagate tau evolution starting from index j+1
        ! call propagate_tau(dm_tl, rho_res, n_tau, dim, j, rho_out)
        ! Step 6: compute result(2:) = Tr(opA * rho_tau)
    end do
    !$omp parallel do private(i, j, k, l, rho_res, tmp, rho_temp)
    do i = 1, n_t
        rho_res = rho_buffer(:,i)
        j = j_array(i)
        do k = 2, n_tau+1
        call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_tl(:,:,j-2+k), dim*dim, &
            rho_res, 1, (0.0d0,0.0d0), rho_temp, 1)
            tmp = reshape(rho_temp, [dim, dim])
            tmp = matmul(opB, tmp)
            result(i,k) = sum([(tmp(l,l), l=1,dim)])
            rho_res = rho_temp
        end do
        write(*,'(A,I5,A)', advance='no') char(13)//'Progress: ', i, ' / '//trim(adjustl(itoa(n_t)))
    end do
    !$omp end parallel do
    write(*,*) char(13)//'Calculation finished.'
    
end subroutine calc_onetime_parallel

subroutine calc_onetime_parallel_block(dm_block, dm_s, rho_init, n_tb, nx_tau, n_map, n_t, n_tfull, dim,&
     opA, opB, opC, time, time_sparse, result)
    use utils
    implicit none
    ! n_tb: number of time points in one timebin
    ! nx_tau: number of timebins for the tau axis
    ! in total the tau axis will have n_tb * nx_tau +1 points
    ! n_map: number of time points in the dm_block
    ! n_t: number of time points in time_sparse, this is the t-axis of the result
    ! n_tfull: number of time points in time, this is the t-axis of the simulation
    integer, intent(in) :: dim, n_tb, nx_tau, n_t, n_tfull, n_map
    complex(8), intent(in) :: dm_block(dim*dim, dim*dim, n_map), dm_s(dim*dim, dim*dim)
    complex(8), intent(in) :: rho_init(dim*dim)
    complex(8), intent(in) :: opA(dim,dim), opB(dim,dim), opC(dim,dim)
    real(8), intent(in) :: time(n_tfull), time_sparse(n_t)
    complex(8), intent(out) :: result(n_t, n_tb*nx_tau+1)

    ! Locals
    integer :: j, i, k, l
    complex(8) :: rho_vec(dim*dim), rho_res(dim*dim)
    complex(8) :: rho_mtx(dim, dim), tmp(dim, dim)
    complex(8) :: rho_temp(dim*dim)
    real(8) :: time_round(n_tfull), time_sparse_round(n_t)

    complex(8) :: rho_buffer(dim*dim, n_t)
    integer :: j_array(n_t)


    rho_vec = rho_init

    ! use dm_block first, then dm_s, 

    ! round time and time_sparse to nearest 1e-6
    time_round = time !nint(time * 1.0d6, kind=8) / 1.0d6
    time_sparse_round = time_sparse !nint(time_sparse * 1.0d6, kind=8) / 1.0d6

    ! Step 1: compute result(1) = Tr(opA * opB * opC * rho_t)
    rho_mtx = reshape(rho_vec, [dim, dim])
    tmp = matmul(opC, rho_mtx)
    tmp = matmul(opB, tmp)
    tmp = matmul(opA, tmp)
    result(1,1) = sum([(tmp(l,l), l=1,dim)])
    j = 1
    do i=1, n_t
        ! Step 2: propagate rho_init up to time(j)
        do while (time_round(j) < time_sparse_round(i))
            if (j <= n_map) then
                call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_block(:,:,j), dim*dim, &
                       rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            else
                call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_s, dim*dim, &
                       rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            end if
            !call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_block(:,:,j), dim*dim, &
            !       rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            rho_vec = rho_res
            j = j + 1
        end do
        
        ! Step 3: compute result(1) = Tr(opA * opB * rho_t), e.g. tau = 0 value
        rho_mtx = reshape(rho_vec, [dim, dim])
        tmp = matmul(opC, rho_mtx)
        tmp = matmul(opB, tmp)
        tmp = matmul(opA, tmp)
        result(i,1) = sum([(tmp(l,l), l=1,dim)])
        ! write(*,*) "tau = 0, time = ", time_sparse_round(i), " result(1) = ", rho_vec(4)
        ! Step 4: apply opC and opA to rho_t and vectorize again
        tmp = matmul(opC, rho_mtx)
        tmp = matmul(tmp, opA)
        rho_res = reshape(tmp, [dim*dim])
        rho_buffer(:,i) = rho_res
        j_array(i) = j
        ! ! Step 5: propagate tau evolution starting from index j+1
        ! call propagate_tau(dm_tl, rho_res, n_tau, dim, j, rho_out)
        ! Step 6: compute result(2:) = Tr(opA * rho_tau)
    end do
    !$omp parallel do private(i, j, k, l, rho_res, tmp, rho_temp)
    do i = 1, n_t
        rho_res = rho_buffer(:,i)
        j = j_array(i)
        ! do m = 1, nx_tau
            do k = 2, nx_tau*n_tb+1
                if (j <= n_map) then
                    call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_block(:,:,j), dim*dim, &
                        rho_res, 1, (0.0d0,0.0d0), rho_temp, 1)
                else
                    call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_s, dim*dim, &
                        rho_res, 1, (0.0d0,0.0d0), rho_temp, 1)
                end if
            !call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_tl(:,:,j-2+k), dim*dim, &
            !    rho_res, 1, (0.0d0,0.0d0), rho_temp, 1)
                tmp = reshape(rho_temp, [dim, dim])
                tmp = matmul(opB, tmp)  ! opB is applied and the result is stored
                result(i,k) = sum([(tmp(l,l), l=1,dim)])
                rho_res = rho_temp
                j = j + 1
                if (j == n_tb + 1) then
                    j = 1 ! reset to the first block
                end if
            end do
        ! end do
        !write(*,'(A,I5,A)', advance='no') char(13)//'Progress: ', i, ' / '//trim(adjustl(itoa(n_t)))
    end do
    !$omp end parallel do
    !write(*,*) char(13)//'Calculation finished.'
    
end subroutine calc_onetime_parallel_block

subroutine calc_single_i(t_axis, t_axis_complete, rho0, dm_sep1, dm_s, dm_tauc2, dim, n_tb,&
    n_map, n_tau, n_t, n_tfull, opA, opB, opC, result, result2)
    use utils
    implicit none
    integer, intent(in) :: n_tb, n_map, n_tau, n_t, n_tfull, dim
    real(8), intent(in) :: t_axis(n_t), t_axis_complete(n_tfull)
    complex(8), intent(in) :: dm_sep1(n_map, dim*dim, dim*dim), dm_s(dim*dim,dim*dim), dm_tauc2(n_map, dim*dim, dim*dim)
    complex(8), intent(in) :: opA(dim,dim), opB(dim,dim), opC(dim,dim)
    complex(8), intent(out) :: result(n_tau)
    complex(8), intent(out) :: result2(n_tau,dim,dim)
    complex(8), intent(in) :: rho0(dim*dim)
    integer :: i, j, k, l
    logical :: use_dm2

    complex(8) :: rho_vec(dim*dim), rho_res(dim*dim)
    complex(8) :: rho_mtx(dim, dim), tmp(dim, dim)


    do j = 1, n_tau
        result2(j,:,:) = reshape(rho_vec, [dim, dim])
    end do

    rho_vec = rho0
    result2(1,:,:) = reshape(rho_vec, [dim, dim])

    i = 1
    do while (t_axis(i) < t_axis_complete(41))
        if (i <= n_map) then
            call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_sep1(i,:,:), dim*dim, &
                   rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
        else
            call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_s, dim*dim, &
                   rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
        end if
        rho_vec = rho_res
        i = i + 1
        result2(i,:,:) = reshape(rho_vec, [dim, dim])
    end do

    rho_mtx = reshape(rho_vec, [dim, dim])
    tmp = matmul(opC, rho_mtx)
    tmp = matmul(opB, tmp)
    tmp = matmul(opA, tmp)
    result(1) = sum([(tmp(l,l), l=1,dim)])

    j = 1
    use_dm2 = .true.  ! only use dm_tauc2 in the first timebin
    do k=2, 2*n_map
        if (j <= n_map) then
            if (use_dm2) then
                call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_tauc2(j,:,:), dim*dim, &
                    rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
            else
                call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_sep1(j,:,:), dim*dim, &
                    rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
            end if
        else
            call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_s, dim*dim, &
                rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
        end if
        rho_vec = rho_res
        result2(i,:,:) = reshape(rho_vec, [dim, dim])
        rho_mtx = reshape(rho_vec, [dim, dim])
        tmp = matmul(opB, result2(i,:,:))
       
        result(k) = sum([(tmp(l,l), l=1,dim)])
        j = j + 1
        i = i + 1
        if (j == n_tb + 1) then
            j = 1 ! reset to the first block
            use_dm2 = .false.
        end if
    end do

end subroutine calc_single_i

subroutine calc_twotime_phonon_block(dm_taucs2, dm_sep1, dm_sep2, dm_s, rho_init, n_tb, nx_tau, n_map, n_t, &
     n_tfull, n_tauc, dim, opA, opB, opC, time, time_sparse, result)
    use utils
    implicit none
    ! n_tb: number of time points in one timebin
    ! nx_tau: number of timebins for the tau axis
    ! in total the tau axis will have n_tb * nx_tau +1 points
    ! n_map: number of time points in the dm_block
    ! n_t: number of time points in time_sparse, this is the t-axis of the result
    ! n_tfull: number of time points in time, this is the t-axis of the simulation
    integer, intent(in) :: dim, n_tb, nx_tau, n_t, n_tfull, n_map, n_tauc
    complex(8), intent(in) :: dm_sep1(dim*dim, dim*dim, n_map), dm_sep2(dim*dim, dim*dim, n_map), dm_s(dim*dim, dim*dim)
    complex(8), intent(in) :: dm_taucs2(dim*dim, dim*dim, n_tauc, n_map) !dm_taucs2(dim*dim, dim*dim, n_tauc, n_map)
    complex(8), intent(in) :: rho_init(dim*dim)
    complex(8), intent(in) :: opA(dim,dim), opB(dim,dim), opC(dim,dim)
    real(8), intent(in) :: time(n_tfull), time_sparse(n_t)
    complex(8), intent(out) :: result(n_t, n_tb*nx_tau+1)
    
    ! Locals
    integer :: j, i, k, l
    complex(8) :: rho_vec(dim*dim), rho_res(dim*dim)
    complex(8) :: rho_mtx(dim, dim), tmp(dim, dim)
    ! complex(8) :: rho_temp(dim*dim)
    real(8) :: time_round(n_tfull), time_sparse_round(n_t)

    complex(8) :: rho_buffer(dim*dim, n_t)
    integer :: j_array(n_t), j_start

    logical :: use_dm2

    !write(*,*) "n_tauc = ", n_tauc, " n_tb = ", n_tb, " nx_tau = ", nx_tau, " n_map = ", n_map

    rho_vec = rho_init

    ! use dm_block first, then dm_s, 

    ! round time and time_sparse to nearest 1e-6
    time_round = time !nint(time * 1.0d6, kind=8) / 1.0d6
    time_sparse_round = time_sparse !nint(time_sparse * 1.0d6, kind=8) / 1.0d6

    ! Step 1: compute result(1) = Tr(opA * opB * opC * rho_t)
    rho_mtx = reshape(rho_vec, [dim, dim])
    tmp = matmul(opB, opC)
    tmp = matmul(opA, tmp)
   ! write(*,*) "opA = ", opA
    !write(*,*) "opB = ", opB
   ! write(*,*) "opC = ", opC
   ! write(*,*) "tmp = ", tmp
   ! write(*,*) "opB*opC", matmul(opB, opC)
    tmp = matmul(tmp, rho_mtx)
    result(1,1) = sum([(tmp(l,l), l=1,dim)])
    j = 1

    do i=1, n_t
        ! Step 2: propagate rho_init up to time(j) using dm_sep1 and the stationary map dm_s 
        do while (time_round(j) < time_sparse_round(i))
            if (j <= n_map) then
                call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_sep1(:,:,j), dim*dim, &
                       rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            else
                call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_s, dim*dim, &
                       rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            end if
            !call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_block(:,:,j), dim*dim, &
            !       rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            rho_vec = rho_res
            j = j + 1
        end do
        
        ! Step 3: compute result(1) = Tr(opA * opB * opC * rho_t), e.g. tau = 0 value
        rho_mtx = reshape(rho_vec, [dim, dim])
        tmp = matmul(opC, rho_mtx)
        tmp = matmul(opB, tmp)
        tmp = matmul(opA, tmp)
        result(i,1) = sum([(tmp(l,l), l=1,dim)])
        ! write(*,*) "tau = 0, time = ", time_sparse_round(i), " result(1) = ", rho_vec(4)
        ! Step 4: apply opB to rho_t and vectorize again
        !tmp = matmul(opC, rho_mtx)
        !tmp = matmul(tmp, opA)
        !rho_res = reshape(tmp, [dim*dim])
        rho_buffer(:,i) = rho_vec
        j_array(i) = j
        ! ! Step 5: propagate tau evolution starting from index j+1
        ! call propagate_tau(dm_tl, rho_res, n_tau, dim, j, rho_out)
        ! Step 6: compute result(2:) = Tr(opA * rho_tau)
    end do

    ! write(*,*) "Finished propagating rho_init up to time(j) using dm_sep1 and dm_s"
    !$omp parallel do private(i, j, j_start, k, l, use_dm2, rho_res, tmp, rho_vec)
    ! use the n_tauc dm_tau2s
    do i = 1, n_tauc
        rho_vec = rho_buffer(:,i)
        j = 1
        j_start = j_array(i)
        use_dm2 = .true.  ! only use dm2 in the first timebin
        do k = 2, nx_tau*n_tb+1
            if (j <= n_map) then
                if (use_dm2) then
                    call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_taucs2(:,:,i,j), dim*dim, &
                        rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
                else
                    call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_sep1(:,:,j), dim*dim, &
                        rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
                end if
            else
                call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_s, dim*dim, &
                    rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
            end if
            rho_vec = rho_res
            tmp = reshape(rho_vec, [dim, dim])
            tmp = matmul(transpose(opB), tmp)  ! opB.T is applied to rho, somewhere it seems the DM is transposed.
            result(i,k) = sum([(tmp(l,l), l=1,dim)])
            j = j + 1
            if (j + j_start == n_tb + 1) then
                j_start = 0 ! reset to the first block
                j = 1 ! reset to the first block
                use_dm2 = .false.
            end if
        end do
    end do
   !$omp end parallel do
    ! use dm_sep2 for the rest of the time points
    !$omp parallel do private(i, j,j_start, k, l, use_dm2, rho_res, tmp, rho_vec)
    do i = n_tauc+1 , n_t
        rho_vec = rho_buffer(:,i)
        j = 1
        j_start = j_array(i) ! we need to start with j=0 because
        ! we need to take all values of the dm maps
        ! but we need to consider as well that the time of the 
        ! propagation is in total t + tau, so we need to take that
        ! t into account, which is at timestep j_array(i)
        ! otherwise the second pulse would be delayed and
        ! always start at tau = tb, not t = tb
        use_dm2 = .true.
        do k = 2, nx_tau*n_tb+1
            if (j <= n_map) then
                if (use_dm2) then
                    call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_sep2(:,:,j), dim*dim, &
                        rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
                else
                    call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_sep1(:,:,j), dim*dim, &
                        rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
                end if
            else
                call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_s, dim*dim, &
                    rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
            end if
            rho_vec = rho_res
            tmp = reshape(rho_vec, [dim, dim])
            tmp = matmul(transpose(opB), tmp)
            result(i,k) = sum([(tmp(l,l), l=1,dim)])
            j = j + 1
            if (j + j_start == n_tb + 1) then
                j_start = 0   ! we only need the delay in the first run
                j = 1 ! reset to the first block
                use_dm2 = .false.
            end if
        end do
        write(*,'(A,I5,A)', advance='no') char(13)//'Progress: ', i, ' / '//trim(adjustl(itoa(n_t)))
    end do
   !$omp end parallel do

end subroutine calc_twotime_phonon_block

subroutine calc_onetime_simple(dm_block, dm_s, rho_init, n_tb, n_map, n_t, n_tfull, dim,&
     opA, opB, opC, time, time_sparse, result)
    use utils
    implicit none
    ! n_tb: number of time points in one timebin
    ! in total the tau axis will have n_tb * nx_tau +1 points
    ! n_map: number of time points in the dm_block
    ! n_t: number of time points in time_sparse, this is the t-axis of the result
    ! n_tfull: number of time points in time, this is the t-axis of the simulation
    integer, intent(in) :: dim, n_tb, n_t, n_tfull, n_map
    complex(8), intent(in) :: dm_block(dim*dim, dim*dim, n_map), dm_s(dim*dim, dim*dim)
    complex(8), intent(in) :: rho_init(dim*dim)
    complex(8), intent(in) :: opA(dim,dim), opB(dim,dim), opC(dim,dim)
    real(8), intent(in) :: time(n_tfull), time_sparse(n_t)
    complex(8), intent(out) :: result(n_t, n_tb+1)

    ! Locals
    integer :: j, i, k, l
    complex(8) :: rho_vec(dim*dim), rho_res(dim*dim)
    complex(8) :: rho_mtx(dim, dim), tmp(dim, dim)
    complex(8) :: rho_temp(dim*dim)
    real(8) :: time_round(n_tfull), time_sparse_round(n_t)

    complex(8) :: rho_buffer(dim*dim, n_t)
    integer :: j_array(n_t)

    rho_vec = rho_init
    ! use dm_block first, then dm_s
    time_round = time
    time_sparse_round = time_sparse
    ! Step 1: compute result(1) = Tr(opA * opB * opC * rho_t), eg. t,tau = 0 value
    rho_mtx = reshape(rho_vec, [dim, dim])
    tmp = matmul(opC, rho_mtx)
    tmp = matmul(opB, tmp)
    tmp = matmul(opA, tmp)
    result(1,1) = sum([(tmp(l,l), l=1,dim)])
    j = 1
    do i=1, n_t
        ! Step 2: propagate rho_init up to time(j)
        do while (time_round(j) < time_sparse_round(i))
            if (j <= n_map) then
                call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_block(:,:,j), dim*dim, &
                       rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            else
                call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_s, dim*dim, &
                       rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            end if
            rho_vec = rho_res
            j = j + 1
        end do
        ! Step 3: compute result(1) = Tr(opA * opB * opC * rho_t), e.g. tau = 0 value
        rho_mtx = reshape(rho_vec, [dim, dim])
        tmp = matmul(opC, rho_mtx)
        tmp = matmul(opB, tmp)
        tmp = matmul(opA, tmp)
        result(i,1) = sum([(tmp(l,l), l=1,dim)])
        ! Step 4: apply opC and opA to rho_t and vectorize again
        tmp = matmul(opC, rho_mtx)
        tmp = matmul(tmp, opA)
        rho_res = reshape(tmp, [dim*dim])
        rho_buffer(:,i) = rho_res
        j_array(i) = j
    end do
    !$omp parallel do private(i, j, k, l, rho_res, tmp, rho_temp)
    do i = 1, n_t
        rho_res = rho_buffer(:,i)
        ! Step 5: propagate tau evolution starting from index j+1
        j = j_array(i)
        ! k loops over the tau points
        do k = 2, n_tb+1
            if (j <= n_map) then
                call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_block(:,:,j), dim*dim, &
                    rho_res, 1, (0.0d0,0.0d0), rho_temp, 1)
            else
                call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_s, dim*dim, &
                    rho_res, 1, (0.0d0,0.0d0), rho_temp, 1)
            end if
            tmp = reshape(rho_temp, [dim, dim])
            tmp = matmul(opB, tmp)  ! opB is applied and the result is stored
            result(i,k) = sum([(tmp(l,l), l=1,dim)])
            rho_res = rho_temp
            j = j + 1
        end do

    end do
    !$omp end parallel do
end subroutine calc_onetime_simple

subroutine calc_onetime_simple_phonon(dm_taucs2, dm_sep1, dm_sep2, dm_s, rho_init, n_tb, n_map, n_t, &
     n_tfull, n_tauc, dim, opA, opB, opC, time, time_sparse, result)
    use utils
    implicit none
    ! n_tb: number of time points in one timebin
    ! nx_tau: number of timebins for the tau axis
    ! in total the tau axis will have n_tb * nx_tau +1 points
    ! n_map: number of time points in the dm_block
    ! n_t: number of time points in time_sparse, this is the t-axis of the result
    ! n_tfull: number of time points in time, this is the t-axis of the simulation
    integer, intent(in) :: dim, n_tb, n_t, n_tfull, n_map, n_tauc
    complex(8), intent(in) :: dm_sep1(dim*dim, dim*dim, n_map), dm_sep2(dim*dim, dim*dim, n_map), dm_s(dim*dim, dim*dim)
    complex(8), intent(in) :: dm_taucs2(dim*dim, dim*dim, n_tauc, n_map) !dm_taucs2(dim*dim, dim*dim, n_tauc, n_map)
    complex(8), intent(in) :: rho_init(dim*dim)
    complex(8), intent(in) :: opA(dim,dim), opB(dim,dim), opC(dim,dim)
    real(8), intent(in) :: time(n_tfull), time_sparse(n_t)
    complex(8), intent(out) :: result(n_t, n_tb+1)
    
    ! Locals
    integer :: j, i, k, l
    complex(8) :: rho_vec(dim*dim), rho_res(dim*dim)
    complex(8) :: rho_mtx(dim, dim), tmp(dim, dim)
    real(8) :: time_round(n_tfull), time_sparse_round(n_t)
    complex(8) :: rho_buffer(dim*dim, n_t)
    integer :: j_array(n_t)
    rho_vec = rho_init

    ! use dm_block first, then dm_s, 
    time_round = time
    time_sparse_round = time_sparse

    ! Step 1: compute result(1) = Tr(opA * opB * opC * rho_t)
    rho_mtx = reshape(rho_vec, [dim, dim])
    tmp = matmul(opB, opC)
    tmp = matmul(opA, tmp)
    tmp = matmul(tmp, rho_mtx)
    result(1,1) = sum([(tmp(l,l), l=1,dim)])
    j = 1

    do i=1, n_t
        ! Step 2: propagate rho_init up to time(j) using dm_sep1 and the stationary map dm_s 
        do while (time_round(j) < time_sparse_round(i))
            if (j <= n_map) then
                call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_sep1(:,:,j), dim*dim, &
                       rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            else
                call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_s, dim*dim, &
                       rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
            end if
            rho_vec = rho_res
            j = j + 1
        end do
        ! Step 3: compute result(1) = Tr(opA * opB * opC * rho_t), e.g. tau = 0 value
        rho_mtx = reshape(rho_vec, [dim, dim])
        tmp = matmul(opC, rho_mtx)
        tmp = matmul(opB, tmp)
        tmp = matmul(opA, tmp)
        result(i,1) = sum([(tmp(l,l), l=1,dim)])
        ! Step 4: apply opB to rho_t and vectorize again
        rho_buffer(:,i) = rho_vec
        j_array(i) = j
        ! Step 5: propagate tau evolution starting from index j+1
        ! Step 6: compute result(2:) = Tr(opA * rho_tau)
    end do

    !$omp parallel do private(i, j, k, l, rho_res, tmp, rho_vec)
    ! use the n_tauc dm_tau2s
    do i = 1, n_tauc
        rho_vec = rho_buffer(:,i)
        j = 1
        do k = 2, n_tb+1
            if (j <= n_map) then
                call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_taucs2(:,:,i,j), dim*dim, &
                        rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
            else
                call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_s, dim*dim, &
                    rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
            end if
            rho_vec = rho_res
            tmp = reshape(rho_vec, [dim, dim])
            tmp = matmul(transpose(opB), tmp)  ! opB.T is applied to rho, somewhere it seems the DM is transposed.
            result(i,k) = sum([(tmp(l,l), l=1,dim)])
            j = j + 1
        end do
    end do
    !$omp end parallel do
    ! use dm_sep2 for the rest of the time points
    !$omp parallel do private(i, j, k, l, rho_res, tmp, rho_vec)
    do i = n_tauc+1 , n_t
        rho_vec = rho_buffer(:,i)
        j = 1
        do k = 2, n_tb+1
            if (j <= n_map) then
                call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_sep2(:,:,j), dim*dim, &
                    rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
            else
                call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_s, dim*dim, &
                    rho_vec, 1, (0.0d0,0.0d0), rho_res, 1)
            end if
            rho_vec = rho_res
            tmp = reshape(rho_vec, [dim, dim])
            tmp = matmul(transpose(opB), tmp)
            result(i,k) = sum([(tmp(l,l), l=1,dim)])
            j = j + 1
        end do
    end do
   !$omp end parallel do
end subroutine calc_onetime_simple_phonon
