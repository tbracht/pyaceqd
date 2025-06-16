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
                tmp = matmul(opB, tmp)
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

subroutine calc_twotime_phonon_block(dm_block, dm_s, rho_init, n_tb, nx_tau, n_map, n_t, n_tfull, dim,&
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
                tmp = matmul(opB, tmp)
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
    
end subroutine calc_twotime_phonon_block

! subroutine calc_onetime_parallel_block_for(dm_block, dm_s, rho_init, n_tb, nx_tau, n_map, n_t, n_tfull, dim,&
!      opA, opB, opC, time, time_sparse, result)
!     use utils
!     implicit none
!     integer, intent(in) :: dim, n_tb, nx_tau, n_t, n_tfull, n_map
!     complex(8), intent(in) :: dm_block(dim*dim, dim*dim, n_map), dm_s(dim*dim, dim*dim)
!     complex(8), intent(in) :: rho_init(dim*dim)
!     complex(8), intent(in) :: opA(dim,dim), opB(dim,dim), opC(dim,dim)
!     real(8), intent(in) :: time(n_tfull), time_sparse(n_t)
!     complex(8), intent(out) :: result(n_t, n_tb*nx_tau+1)

!     ! Locals
!     integer :: j, i, k, l
!     complex(8) :: rho_vec(dim*dim), rho_res(dim*dim)
!     complex(8) :: rho_mtx(dim, dim), tmp(dim, dim)
!     complex(8) :: rho_temp(dim*dim)
!     real(8) :: time_round(n_tfull), time_sparse_round(n_t)
!     integer :: mat_idx(n_tfull)

!     complex(8) :: rho_buffer(dim*dim, n_t)
!     integer :: j_array(n_t)

!     do i = 1, n_tfull
!         if (i <= n_map) then
!             mat_idx(i) = i
!         else
!             mat_idx(i) = n_map ! use dm_s for the rest
!         end if
!     end do

!     rho_vec = rho_init

!     ! use dm_block first, then dm_s, 

!     ! round time and time_sparse to nearest 1e-6
!     time_round = time !nint(time * 1.0d6, kind=8) / 1.0d6
!     time_sparse_round = time_sparse !nint(time_sparse * 1.0d6, kind=8) / 1.0d6

!     ! Step 1: compute result(1) = Tr(opA * opB * opC * rho_t)
!     rho_mtx = reshape(rho_vec, [dim, dim])
!     tmp = matmul(opC, rho_mtx)
!     tmp = matmul(opB, tmp)
!     tmp = matmul(opA, tmp)
!     result(1,1) = sum([(tmp(l,l), l=1,dim)])
!     j = 1

!     do i=1, n_t
!         ! Step 2: propagate rho_init up to time(j)
!         do while (time_round(j) < time_sparse_round(i))
!             call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), dm_block(:,:,mat_idx(j)), dim*dim, &
!                   rho_vec, 1, (0.0d0, 0.0d0), rho_res, 1)
!             rho_vec = rho_res
!             j = j + 1
!         end do
        
!         ! Step 3: compute result(1) = Tr(opA * opB * rho_t), e.g. tau = 0 value
!         rho_mtx = reshape(rho_vec, [dim, dim])
!         tmp = matmul(opC, rho_mtx)
!         tmp = matmul(opB, tmp)
!         tmp = matmul(opA, tmp)
!         result(i,1) = sum([(tmp(l,l), l=1,dim)])
!         ! write(*,*) "tau = 0, time = ", time_sparse_round(i), " result(1) = ", rho_vec(4)
!         ! Step 4: apply opB to rho_t and vectorize again
!         tmp = matmul(opC, rho_mtx)
!         tmp = matmul(tmp, opA)
!         rho_res = reshape(tmp, [dim*dim])
!         rho_buffer(:,i) = rho_res
!         j_array(i) = j
!         ! ! Step 5: propagate tau evolution starting from index j+1
!         ! call propagate_tau(dm_tl, rho_res, n_tau, dim, j, rho_out)
!         ! Step 6: compute result(2:) = Tr(opA * rho_tau)
!     end do
!     !$omp parallel do private(i, j, k, l, rho_res, tmp, rho_temp)
!     do i = 1, n_t
!         rho_res = rho_buffer(:,i)
!         j = j_array(i)
!         ! do m = 1, nx_tau
!             do k = 2, nx_tau*n_tb+1
!                 ! if (j <= n_map) then
!                 !     call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_block(:,:,j), dim*dim, &
!                 !         rho_res, 1, (0.0d0,0.0d0), rho_temp, 1)
!                 ! else
!                 !     call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_s, dim*dim, &
!                 !         rho_res, 1, (0.0d0,0.0d0), rho_temp, 1)
!                 ! end if
!                 call zgemv('N', dim*dim, dim*dim, (1.0d0,0.0d0), dm_block(:,:,mat_idx(mod(j,n_tb))),&
!                  dim*dim, rho_res, 1, (0.0d0,0.0d0), rho_temp, 1)
!                 tmp = reshape(rho_temp, [dim, dim])
!                 tmp = matmul(opB, tmp)
!                 result(i,k) = sum([(tmp(l,l), l=1,dim)])
!                 rho_res = rho_temp
!                 j = j + 1
!                 ! if (j == n_tb + 1) then
!                 !     j = 1 ! reset to the first block
!                 ! end if
!             end do
!         ! end do
!         write(*,'(A,I5,A)', advance='no') char(13)//'Progress: ', i, ' / '//trim(adjustl(itoa(n_t)))
!     end do
!     !$omp end parallel do
!     write(*,*) char(13)//'Calculation finished.        '
    
! end subroutine calc_onetime_parallel_block_for