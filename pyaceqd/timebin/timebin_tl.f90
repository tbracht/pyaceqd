! f2py -c --f90flags="-fopenmp" -m timebin_tl timebin_tl.f90 -lopenblas -lgomp

module utils
contains
    function itoa(i) result(str)
        integer, intent(in) :: i
        character(len=20) :: str
        write(str, '(I0)') i
    end function itoa

    ! Round to 6 decimal places using multiplication and division
! This avoids floating-point comparison issues
    function round_to_6(x) result(rounded)
        implicit none
        real(8), intent(in) :: x
        real(8) :: rounded
        integer(8) :: scale
        scale = 1000000_8  ! 10^6
        rounded = real(nint(x * scale, kind=8), kind=8) / scale
    end function round_to_6 

    ! filepath: /home/thomas/Documents/pyaceqd/pyaceqd/timebin/timebin_tl.f90
    function fast_propagate(rho, dm_tl_precalc, n_steps, dimsquare, n_precalc) result(rho_out)
        implicit none
        integer, intent(in) :: n_steps, dimsquare, n_precalc
        complex(8), intent(in) :: rho(dimsquare)
        complex(8), intent(in) :: dm_tl_precalc(dimsquare, dimsquare, n_precalc)
        complex(8) :: rho_out(dimsquare), rho_temp(dimsquare)
        integer :: i, n_tmp

        rho_out = rho
        n_tmp = n_steps

        ! Loop through bits, starting from least significant
        i = 0
        do while (n_tmp > 0)
            if (iand(n_tmp, 1) == 1) then
                ! If bit is 1, multiply with corresponding power
                call zgemv('N', dimsquare, dimsquare, (1.0d0, 0.0d0), &
                          dm_tl_precalc(:,:,i+1), dimsquare, &
                          rho_out, 1, (0.0d0, 0.0d0), rho_temp, 1)
                rho_out = rho_temp
            end if
            n_tmp = ishft(n_tmp, -1)  ! Shift right by 1 bit
            i = i + 1
        end do
    end function fast_propagate

    ! ideas: test these functions in the module directly from python
    function propagate_tb(t_start, t_stop, dt, rho, dm_tl, dm_tl_precalc, n_precalc, dimsquare, n_dm) result(rho_out)
        implicit none
        integer, intent(in) :: dimsquare, n_dm, n_precalc
        real(8), intent(in) :: t_start, t_stop, dt
        complex(8), intent(in) :: rho(dimsquare), dm_tl(dimsquare, dimsquare, n_dm), dm_tl_precalc(dimsquare, dimsquare, n_precalc)
        complex(8) :: rho_out(dimsquare)
        integer :: n_steps, steps_dm, n_start, n_stop
        complex(8) :: rho_temp(dimsquare)
        
        n_start = int(round_to_6(t_start) / dt)
        n_stop = int(round_to_6(t_stop) / dt)
        n_steps = n_stop - n_start
        steps_dm = min(n_dm - n_start, n_steps)
        rho_temp = rho
        !write(*,*) "from", t_start, "to", t_stop, "in", n_steps, "steps, n_start", n_start, "with", steps_dm, "dm_steps"
        do while (steps_dm > 0)
            call zgemv('N', dimsquare, dimsquare, (1.0d0, 0.0d0), dm_tl(:,:,n_start+1), dimsquare, &
                 rho_temp, 1, (0.0d0, 0.0d0), rho_out, 1)
            rho_temp = rho_out
            n_steps = n_steps - 1
            n_start = n_start + 1
            steps_dm = steps_dm - 1
        end do
        if (n_steps > 0) then
            rho_temp = fast_propagate(rho_temp, dm_tl_precalc, n_steps, dimsquare, n_precalc)
        end if
        rho_out = rho_temp
    end function propagate_tb

    function apply_matrix_from_right(rho, op, dim) result(rho_out)
        implicit none
        integer, intent(in) :: dim
        complex(8), intent(in) :: rho(dim, dim), op(dim, dim)
        complex(8) :: rho_out(dim, dim)
        ! complex(8) :: rho_mtx(dim, dim)!, rho_temp(dim, dim)
        !rho_mtx = reshape(rho, [dim, dim])
        rho_out = matmul(rho, op)
        !call zgemm('N', 'N', dim, dim, dim, (1.0d0, 0.0d0), &
         !    rho, dim, (op), dim, (0.0d0, 0.0d0), (rho_out), dim)
        !rho_out = reshape(rho_mtx, [dim*dim])
    end function apply_matrix_from_right

    function apply_matrix_from_left(rho, op, dim) result(rho_out)
        implicit none
        integer, intent(in) :: dim
        complex(8), intent(in) :: rho(dim,dim), op(dim, dim)
        complex(8) :: rho_out(dim, dim)
        ! complex(8) :: rho_mtx(dim, dim)
        rho_out = matmul(op, rho)
    end function apply_matrix_from_left

    function apply_left(rho, op, dim) result(rho_out)
        implicit none
        integer, intent(in) :: dim
        complex(8), intent(in) :: rho(dim*dim), op(dim,dim)
        complex(8) :: rho_out(dim*dim)
        complex(8) :: rho_mtx(dim, dim)
        rho_mtx = reshape(rho, [dim, dim])
        rho_mtx = matmul(op, rho_mtx)
        rho_out = reshape(rho_mtx, [dim*dim])
    end function apply_left

    function apply_right(rho, op, dim) result(rho_out)
        implicit none
        integer, intent(in) :: dim
        complex(8), intent(in) :: rho(dim*dim), op(dim,dim)
        complex(8) :: rho_out(dim*dim)
        complex(8) :: rho_mtx(dim, dim)
        rho_mtx = reshape(rho, [dim, dim])
        rho_mtx = matmul(rho_mtx, op)
        rho_out = reshape(rho_mtx, [dim*dim])
    end function apply_right

    function apply_operator(rho, op, dim) result(rho_out)
        implicit none
        integer, intent(in) :: dim
        complex(8), intent(in) :: rho(dim*dim), op(dim*dim, dim*dim)
        complex(8) :: rho_out(dim*dim)

        call zgemv('N', dim*dim, dim*dim, (1.0d0, 0.0d0), &
             op, dim*dim, rho, 1, (0.0d0, 0.0d0), rho_out, 1)
    end function apply_operator

    function test_reshape(rho, dim) result(rho_out)
        implicit none
        integer, intent(in) :: dim
        complex(8), intent(in) :: rho(dim,dim)
        complex(8) :: rho_out(dim,dim)

        !rho_out = reshape(transpose(rho), [dim*dim])
        rho_out = transpose(reshape(reshape(transpose(rho), [dim*dim]), [dim, dim]))
    end function test_reshape

end module utils

subroutine four_time(dm_1, dm_2, rho_init, t1, precalc_tls, n_t, dt, n_map, dim,&
     op_1, op_2, op_3, op_4, tb, n_precalc, result)!, rho_buffer)
    use utils
    implicit none
    ! n_tb: number of time points in one timebin
    ! nx_tau: number of timebins for the tau axis
    ! in total the tau axis will have n_tb * nx_tau +1 points
    ! n_map: number of time points in the dm_block
    ! n_t: number of time points in time_sparse, this is the t-axis of the result
    ! n_tfull: number of time points in time, this is the t-axis of the simulation
    integer, intent(in) :: dim, n_t, n_map, n_precalc
    complex(8), intent(in) :: dm_1(dim*dim, dim*dim, n_map), dm_2(dim*dim, dim*dim, n_map), precalc_tls(dim*dim, dim*dim, n_precalc)
    complex(8), intent(in) :: rho_init(dim*dim)
    complex(8), intent(in) :: op_1(dim, dim), op_2(dim, dim), op_3(dim, dim), op_4(dim, dim)
    real(8), intent(in) :: t1(0:n_t-1), dt, tb
    complex(8), intent(out) :: result(0:n_t-1, 0:n_t-1)!, rho_buffer(dim*dim, 0:n_t-1)

    ! Locals
    integer :: j, i, l
    complex(8) :: rho_vec(dim*dim), rho_res(dim*dim), rho_temp(dim*dim)
    complex(8) :: rho_mtx(dim, dim)
    real(8) :: t1_now, t2, t1round(0:n_t-1) !, time(0:n_t-1)

    result = (0.0d0, 0.0d0)
    ! write(*,*) rho_init
    ! rho_vec = reshape(rho_init, [dim*dim])
    ! use dm_block first, then dm_s, 

    ! round time and time_sparse to nearest 1e-6
    ! time_round = nint(time * 1.0d6, kind=8) / 1.0d6
    !time = t1  !nint(t1 * 1.0d6, kind=8) / 1.0d6  ! ensure t1 is rounded to 1e-6
    ! t1round = nint(t1 * 1.0d6, kind=8) / 1.0d6

    ! Step 1: compute result(1) = Tr(opA * opB * opC * rho_t)
    
    !$omp parallel do private(i, j, l, t2, t1_now, rho_res, rho_mtx, rho_vec)
    do i=0, n_t-1
        t1_now = t1(i)
        ! propagate to t1
        rho_vec = propagate_tb(0.0d0, t1_now, dt, rho_init, dm_1, precalc_tls, n_precalc, dim*dim, n_map)
        !rho_buffer(:,i) = rho_vec
        do j=0, n_t-1-i
            t2 = t1(i + j)
            ! if (i == 10) then
            !     write(*,*) "i,j:", i, j, " t1,t2:", t1_now, t2
            ! end if 
            rho_res = rho_vec
            ! apply op1 from right
            rho_res = apply_right(rho_res, op_1, dim)
            ! propagate to t2
            rho_res = propagate_tb(t1_now, t2, dt, rho_res, dm_1, precalc_tls, n_precalc, dim*dim, n_map)
            ! apply op2 from right
            rho_res = apply_right(rho_res, op_2, dim)
            ! propagate to t1+tb
            ! first from t2 to tb
            rho_res = propagate_tb(t2, tb, dt, rho_res, dm_1, precalc_tls, n_precalc, dim*dim, n_map)
            ! then from tb to t1+tb
            rho_res = propagate_tb(0.0d0, t1_now, dt, rho_res, dm_2, precalc_tls, n_precalc, dim*dim, n_map)
            ! apply op3 from left at t1+tb
            rho_res = apply_left(rho_res, op_3, dim)
            ! propagate to t2+tb
            rho_res = propagate_tb(t1_now, t2, dt, rho_res, dm_2, precalc_tls, n_precalc, dim*dim, n_map)
            ! apply op4 from left
            rho_res = apply_left(rho_res, op_4, dim)
            rho_mtx = reshape(rho_res, [dim, dim])
            result(i,j+i) = sum([(rho_mtx(l,l), l=1,dim)])
        end do
    end do
    !$omp end parallel do
end subroutine four_time

subroutine four_time_8op(dm_1, dm_2, rho_init, t1, precalc_tls, n_t, dt, n_map, dim,&
    op_et1l, op_et1r, op_et2l, op_et2r,&
    op_lt1l, op_lt1r, op_lt2l, op_lt2r,&
    early_only, late_t1_only, tb, n_precalc, result)!, rho_buffer)
    use utils
    implicit none
    ! n_tb: number of time points in one timebin
    ! nx_tau: number of timebins for the tau axis
    ! in total the tau axis will have n_tb * nx_tau +1 points
    ! n_map: number of time points in the dm_block
    ! n_t: number of time points in time_sparse, this is the t-axis of the result
    ! n_tfull: number of time points in time, this is the t-axis of the simulation
    integer, intent(in) :: dim, n_t, n_map, n_precalc
    logical, intent(in) :: early_only, late_t1_only
    complex(8), intent(in) :: dm_1(dim*dim, dim*dim, n_map), dm_2(dim*dim, dim*dim, n_map), precalc_tls(dim*dim, dim*dim, n_precalc)
    complex(8), intent(in) :: rho_init(dim*dim)
    complex(8), intent(in) :: op_et1l(dim, dim), op_et1r(dim, dim), op_et2l(dim, dim), op_et2r(dim, dim)
    complex(8), intent(in) :: op_lt1l(dim, dim), op_lt1r(dim, dim), op_lt2l(dim, dim), op_lt2r(dim, dim)
    real(8), intent(in) :: t1(0:n_t-1), dt, tb
    complex(8), intent(out) :: result(0:n_t-1, 0:n_t-1)!, rho_buffer(dim*dim, 0:n_t-1)

    ! Locals
    integer :: j, i, l
    complex(8) :: rho_vec(dim*dim), rho_res(dim*dim) !, rho_temp(dim*dim)
    complex(8) :: rho_mtx(dim, dim)
    real(8) :: t1_now, t2 !, t1round(0:n_t-1) !, time(0:n_t-1)

    result = (0.0d0, 0.0d0)
    ! write(*,*) rho_init
    ! rho_vec = reshape(rho_init, [dim*dim])
    ! use dm_block first, then dm_s, 

    ! round time and time_sparse to nearest 1e-6
    ! time_round = nint(time * 1.0d6, kind=8) / 1.0d6
    !time = t1  !nint(t1 * 1.0d6, kind=8) / 1.0d6  ! ensure t1 is rounded to 1e-6
    ! t1round = nint(t1 * 1.0d6, kind=8) / 1.0d6

    ! Step 1: compute result(1) = Tr(opA * opB * opC * rho_t)
    
    !$omp parallel do private(i, j, l, t2, t1_now, rho_res, rho_mtx, rho_vec)
    do i=0, n_t-1
        t1_now = t1(i)
        ! propagate to t1
        rho_vec = propagate_tb(0.0d0, t1_now, dt, rho_init, dm_1, precalc_tls, n_precalc, dim*dim, n_map)
        !rho_buffer(:,i) = rho_vec
        do j=0, n_t-1-i
            t2 = t1(i + j)
            ! if (i == 10) then
            !     write(*,*) "i,j:", i, j, " t1,t2:", t1_now, t2
            ! end if 
            rho_res = rho_vec
            ! apply op1 from right
            rho_res = apply_right(rho_res, op_et1r, dim)
            rho_res = apply_left(rho_res, op_et1l, dim)
            ! propagate to t2
            rho_res = propagate_tb(t1_now, t2, dt, rho_res, dm_1, precalc_tls, n_precalc, dim*dim, n_map)
            ! apply op2 from right
            rho_res = apply_right(rho_res, op_et2r, dim)
            rho_res = apply_left(rho_res, op_et2l, dim)
            if (early_only) then
                rho_mtx = reshape(rho_res, [dim, dim])
                result(i,j+i) = sum([(rho_mtx(l,l), l=1,dim)])
                cycle  ! skip the rest
            end if
            ! propagate to t1+tb
            ! first from t2 to tb
            rho_res = propagate_tb(t2, tb, dt, rho_res, dm_1, precalc_tls, n_precalc, dim*dim, n_map)
            ! then from tb to t1+tb
            rho_res = propagate_tb(0.0d0, t1_now, dt, rho_res, dm_2, precalc_tls, n_precalc, dim*dim, n_map)
            ! apply op3 from left at t1+tb
            rho_res = apply_right(rho_res, op_lt1r, dim)
            rho_res = apply_left(rho_res, op_lt1l, dim)
            if (late_t1_only) then
                rho_mtx = reshape(rho_res, [dim, dim])
                result(i,j+i) = sum([(rho_mtx(l,l), l=1,dim)])
                cycle  ! skip the rest
            end if
            ! propagate to t2+tb
            rho_res = propagate_tb(t1_now, t2, dt, rho_res, dm_2, precalc_tls, n_precalc, dim*dim, n_map)
            ! apply op4 from left
            rho_res = apply_right(rho_res, op_lt2r, dim)
            rho_res = apply_left(rho_res, op_lt2l, dim)
            rho_mtx = reshape(rho_res, [dim, dim])
            result(i,j+i) = sum([(rho_mtx(l,l), l=1,dim)])
        end do
    end do
    !$omp end parallel do
end subroutine four_time_8op

subroutine dynamics_t1(dm_1, dm_2, rho_init, t1, precalc_tls, n_t, dt, n_map, dim,&
    tb, n_precalc, result)
    use utils
    implicit none
    ! n_tb: number of time points in one timebin
    ! nx_tau: number of timebins for the tau axis
    ! in total the tau axis will have n_tb * nx_tau +1 points
    ! n_map: number of time points in the dm_block
    ! n_t: number of time points in time_sparse, this is the t-axis of the result
    ! n_tfull: number of time points in time, this is the t-axis of the simulation
    integer, intent(in) :: dim, n_t, n_map, n_precalc
    complex(8), intent(in) :: dm_1(dim*dim, dim*dim, n_map), dm_2(dim*dim, dim*dim, n_map), precalc_tls(dim*dim, dim*dim, n_precalc)
    complex(8), intent(in) :: rho_init(dim*dim)
    ! complex(8), intent(in) :: op_1(dim*dim, dim*dim), op_2(dim*dim, dim*dim), op_3(dim*dim, dim*dim), op_4(dim*dim, dim*dim)
    real(8), intent(in) :: t1(0:n_t-1), dt, tb
    complex(8), intent(out) :: result(dim*dim, 0:2*n_t-2)

    ! Locals
    integer :: j, i, l
    complex(8) :: rho_vec(dim*dim), rho_res(dim*dim), rho_temp(dim*dim)
    complex(8) :: rho_mtx(dim, dim)
    real(8) :: tnow,tnext !, time(0:n_t-1)

    result(:,0) = rho_init
    !do i=0,5
    !    write(*,*) i 
    !end do
    do i = 0, n_t-2
       tnow = t1(i)
       tnext = t1(i+1)
       result(:,i+1) = propagate_tb(tnow, tnext, dt, result(:,i), dm_1, precalc_tls, n_precalc, dim*dim, n_map)
    end do
    do i = 0, n_t-2
       tnow = t1(i)
       tnext = t1(i+1)
       result(:,i+1+n_t-1) = propagate_tb(tnow, tnext, dt, result(:,i+n_t-1), dm_2, precalc_tls, n_precalc, dim*dim, n_map)
    end do
end subroutine dynamics_t1

subroutine dynamics_t1_t2(dm_1, dm_2, t1op, t2op, rho_init, t1, precalc_tls, n_t, dt, n_map, dim,&
    tb, op_1, op_2, op_3, n_precalc, result)
    use utils
    implicit none
    ! n_tb: number of time points in one timebin
    ! nx_tau: number of timebins for the tau axis
    ! in total the tau axis will have n_tb * nx_tau +1 points
    ! n_map: number of time points in the dm_block
    ! n_t: number of time points in time_sparse, this is the t-axis of the result
    ! n_tfull: number of time points in time, this is the t-axis of the simulation
    integer, intent(in) :: dim, n_t, n_map, n_precalc
    complex(8), intent(in) :: dm_1(dim*dim, dim*dim, n_map), dm_2(dim*dim, dim*dim, n_map), precalc_tls(dim*dim, dim*dim, n_precalc)
    complex(8), intent(in) :: rho_init(dim*dim)
    !complex(8), intent(in) :: op_1(dim*dim, dim*dim), op_2(dim*dim, dim*dim), op_3(dim*dim, dim*dim) !, op_4(dim*dim, dim*dim)
    complex(8), intent(in) :: op_1(dim,dim), op_2(dim,dim), op_3(dim,dim) !, op_4(dim*dim, dim*dim)
    
    real(8), intent(in) :: t1(0:n_t-1), dt, tb, t1op, t2op
    complex(8), intent(out) :: result(dim*dim, 0:2*n_t-2)

    ! Locals
    integer :: i
    complex(8) :: rho_vec(dim*dim), rho_res(dim*dim), rho_temp(dim*dim)
    complex(8) :: rho_mtx(dim, dim)
    real(8) :: tnow,tnext !, time(0:n_t-1)

    result(:,0) = rho_init
    !do i=0,5
    !    write(*,*) i 
    !end do
    do i = 0, n_t-2
        tnow = t1(i)
        tnext = t1(i+1)
        rho_temp = result(:,i)
        if (tnow == t1op) then
            write(*,*) 'applying op1 at time', tnow, t1op
            rho_temp = apply_right(result(:,i), op_1, dim)
        end if
        if (tnow == t2op) then
            write(*,*) 'applying op2 at time', tnow, t2op
            rho_temp = apply_right(result(:,i), op_2, dim)
        end if
        result(:,i+1) = propagate_tb(tnow, tnext, dt, rho_temp, dm_1, precalc_tls, n_precalc, dim*dim, n_map)
    end do
    do i = 0, n_t-2
        tnow = t1(i)
        tnext = t1(i+1)
        rho_temp = result(:,i+n_t-1)
        if (tnow == t1op) then
            write(*,*) 'applying op3 at time', tnow+tb, t1op+tb
            rho_temp = apply_left(result(:,i+n_t-1), op_3, dim)
        end if
        result(:,i+1+n_t-1) = propagate_tb(tnow, tnext, dt, rho_temp, dm_2, precalc_tls, n_precalc, dim*dim, n_map)
    end do
end subroutine dynamics_t1_t2
