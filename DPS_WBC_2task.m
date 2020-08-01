T = 1000;
q_i = [0.0; 0.0; 0.0; pi/16];

global rho1;
global rho2;
global rho3;
global h;
global Q;
global R;

rho1 = 0.6;
rho2 = 0.6;
rho3 = 0.6;
h = 0.4;
Q = eye(4);
R = eye(8);
A = eye(4);


%Initial Conditions
[x1_i, y1_i, x2_i, y2_i, x3_i, y3_i, x4_i, y4_i] = calc_joints(q_i)

%Define Nominal Trajectory
xt1_nom_traj = linspace(x1_i,2.25,T);
yt1_nom_traj = linspace(y1_i,0.7,T);
xt2_nom_traj = linspace(x4_i,0.5,T);
yt2_nom_traj = linspace(y4_i,y4_i,T);
nom_traj = [xt1_nom_traj; yt1_nom_traj; xt2_nom_traj; yt2_nom_traj];


%% PROPAGATE NOMINAL TRAJECTORY
qt_nom = q_i;

[J1, J2, J3, J4] = calc_J(q_i);
Jt1 = J1;
Jt2 = J4;
N1 = eye(4)-pinv(Jt1)*Jt1;
B = [Jt1 zeros(2,4); Jt2 Jt2*N1];

joint_pos_init = [x1_i; y1_i; x2_i; y2_i; x3_i; y3_i; x4_i; y4_i];
nom_traj_t = [x1_i; y1_i; x4_i; y4_i];

B_hist(:,:,1) = B;
joint_pos_hist(:,1) = joint_pos_init;

for t=2:T
    %Find Nominal Joint Perturbations
    dxt_nom = nom_traj(:,t)-nom_traj_t;
    dq_nom = pinv(B)*dxt_nom;
    
    dq1 = dq_nom(1:4);
    dq2 = dq_nom(5:8);
    dq = dq1 + N1*dq2;
    
    %Joint Angle at Next Time Step
    qt_nom = qt_nom + dq;
    
    %Redefine Joint Positions
    [x1t, y1t, x2t, y2t, x3t, y3t, x4t, y4t] = calc_joints(qt_nom)
    
    joint_pos_t = [x1t; y1t; x2t; y2t; x3t; y3t; x4t; y4t];
    nom_traj_t = [x1t; y1t; x4t; y4t];
    
    %Recompute Jacobians and B Matrix
    [J1, J2, J3, J4] = calc_J(q_i);
    Jt1 = J1;
    Jt2 = J4;
    N1 = eye(4)-pinv(Jt1)*Jt1;
    B = [Jt1 zeros(2,4); Jt2 Jt2*N1];
    B_hist(:,:,t) = B;
    
    joint_pos_hist(:,t) = joint_pos_t;
end

figure(1)
plot(joint_pos_hist(1,:),joint_pos_hist(2,:),'.')
hold on
plot(joint_pos_hist(3,:),joint_pos_hist(4,:),'.')
hold on
plot(joint_pos_hist(5,:),joint_pos_hist(6,:),'.')
hold on
plot(joint_pos_hist(7,:),joint_pos_hist(8,:),'.')
title('Nominal Joint Trajectories LQR')
xlabel('xpos (m)')
ylabel('ypos (m)')
legend('End Effector 1','End Effector 2','End Effector 3','End Effector 4')


%% Calculate Recatti Recursion With Sparsity
for t = linspace(T-1,1,T-1)
    XT1 = eye(2);
    XT2 = eye(4);
    Q1 = Q(3:4,3:4);
    Q2 = Q;
    R1 = R(5:8,5:8);
    R2 = R;
    A1 = A(3:4,3:4);
    A2 = A;
    B1 = B_hist(3:4,5:8,t);
    B2 = B_hist(:,:,t);
 
    if t == T-1
        om1t = R1+B1'*XT1*B1;
        K1t = -inv(om1t)*(A1'*XT1*B1)';
        X1t = Q1+A1'*XT1*A1-K1t'*om1t*K1t;
        
        om2t = R2+B2'*XT2*B2;
        K2t = -inv(om2t)*(A2'*XT2*B2)'
        X2t = Q2+A2'*XT2*A2-K2t'*om2t*K2t;
        
        K1_hist(:,:,t) = K1t;
        K2_hist(:,:,t) = K2t;
        
    else
        om1t = R1+B1'*X1t*B1;
        K1t = -inv(om1t)*(A1'*X1t*B1)';
        X1t = Q1+A1'*X1t*A1-K1t'*om1t*K1t;
        
        om2t = R2+B2'*X2t*B2;
        K2t = -inv(om2t)*(A2'*X2t*B2)';
        X2t = Q2+A2'*X2t*A2-K2t'*om2t*K2t;
        
        K1_hist(:,:,t) = K1t;
        K2_hist(:,:,t) = K2t;
        
    end
end

%% Run Simulation with Sparsity
xt_error = [0; 0; 0; 0];

eta1 = [0; 0];
eta2 = [0; 0; 0; 0];
I = eye(8);

for t=1:T-1
    eta1tp1 = [0.01*randn(1,1); 0];
    eta2tp1 = [0.01*randn(3,1); 0];
    
    x_error_hist(:,t) = xt_error;
    
    ut = I(:,5:8)*K1_hist(:,:,t)*eta1+I*K2_hist(:,:,t)*eta2;
    xtp1_error = A*xt_error + B_hist(:,:,t)*ut;

    eta1 = eta1tp1;
    eta2 = eta2tp1;
    
    xt_error = xtp1_error;
    u_hist(:,t) = ut;
    

    cost = xt_error'*Q*xt_error+ut'*R*ut;
    J_hist(t) = cost;

end

x_actual_hist = [joint_pos_hist(1:2,:); joint_pos_hist(7:8,:)] - x_error_hist;

%% Plot Results
figure(2)
plot([1:T],x_error_hist)
title('Task Error LQR')
xlabel('Timestep')
ylabel('distance (m)')
legend('Task 1 xpos','Task 1 ypos','Task 2 xpos','Task 2 ypos')

figure(3)
plot(x_actual_hist(1,:),x_actual_hist(2,:),'-o')
hold on
plot(x_actual_hist(3,:),x_actual_hist(4,:),'-o')
hold on
plot(joint_pos_hist(1,:),joint_pos_hist(2,:),'.','LineWidth',1)
hold on
plot(joint_pos_hist(7,:),joint_pos_hist(8,:),'.','LineWidth',1)
title('Actual Task Trajectories')
xlabel('xpos (m)')
ylabel('ypos (m)')
legend('q4 actual','q1 actual','q4 nominal','q1 nominal')


figure(4)
plot([1:T],u_hist)
title('Joint Input (dq) Over Time LQR')
xlabel('Timestep')
ylabel('Joint Angle (degrees)')
legend('q4','q3','q2','q1')

figure(5)
plot([1:T],J_hist)
title('Average cost')
xlabel('Timestep')
ylabel('Cost')


%% Function For Jacobians
function [J1, J2, J3, J4] = calc_J(q_t)
    rho1 = 0.6;
    rho2 = 0.6;
    rho3 = 0.6;
    h = 0.4;

    th0 = q_t(1);
    th1 = q_t(2);
    th2 = q_t(3);
    th3 = q_t(4);

    J1 = [1 -rho1*sin(th1)-rho2*sin(th1+th2)-rho3*sin(th1+th2+th3) -rho2*sin(th1+th2)-rho3*sin(th1+th2+th3) -rho3*sin(th1+th2+th3);
        0 rho1*cos(th1)+rho2*cos(th1+th2)+rho3*cos(th1+th2+th3) rho2*cos(th1+th2)+rho3*cos(th1+th2+th3) rho3*cos(th1+th2+th3)];
    
    J2 = [1 -rho1*sin(th1)-rho2*sin(th1+th2) -rho2*sin(th1+th2) 0;
        0 rho1*cos(th1)+rho2*cos(th1+th2) rho2*cos(th1+th2) 0];
    
    J3 = [1 -rho1*sin(th1) 0 0;
        0 rho1*cos(th1) 0 0];
    
    J4 = [1 0 0 0; 0 0 0 0];
end


%% Function For Joint Positions
function [x1, y1, x2, y2, x3, y3, x4, y4] = calc_joints(q_t)
    rho1 = 0.6;
    rho2 = 0.6;
    rho3 = 0.6;
    h = 0.4;

    th0 = q_t(1);
    th1 = q_t(2);
    th2 = q_t(3);
    th3 = q_t(4);

    x1 = th0+rho1*cos(th1)+rho2*cos(th1+th2)+rho3*cos(th1+th2+th3);
    y1 = h+rho1*sin(th1)+rho2*sin(th1+th2)+rho3*sin(th1+th2+th3);

    x2 = th0+rho1*cos(th1)+rho2*cos(th1+th2);
    y2 = h+rho1*sin(th1)+rho2*sin(th1+th2);

    x3 = th0+rho1*cos(th1);
    y3 = h+rho1*sin(th1);

    x4 = th0;
    y4 = h/2;
end