clear all; close all; clc
num_inputs = 500;
% t = 0:0.1:19;
time_steps = 20;%size(t,2);%20;
% ts = 39;
resp_all = zeros(num_inputs*time_steps,9);
% resp_all(:,:,3) = 1;
uk = zeros(num_inputs,2);
% % multi dir, multi dist 
% % one dir - 45 degree
% uk(:,1) = rand(num_inputs,1);
% uk(:,2) = uk(:,1);
% % two dir - 45 + 325 degree
% uk(1:num_inputs/2,1) = rand(num_inputs/2,1);
% uk(1:num_inputs/2,2) = uk(1:num_inputs/2,1);
% uk(num_inputs/2+1:num_inputs,1) = rand(num_inputs/2,1);
% uk(num_inputs/2+1:num_inputs,2) = -1*uk(num_inputs/2+1:num_inputs,1);
% % four dir - 45 + 135 + 225 +325 degree
% uk(1:num_inputs/2,1) = -1+(1+1).*rand(num_inputs/2,1);
% uk(1:num_inputs/2,2) = uk(1:num_inputs/2,1);
% uk(num_inputs/2+1:num_inputs,1) = -1+(1+1).*rand(num_inputs/2,1);
% uk(num_inputs/2+1:num_inputs,2) = -1*uk(num_inputs/2+1:num_inputs,1);
% eight dir - 0 + 45 + 90 + 135 + 180 + 225 + 270 +325 degree
% % 0 and 180
% uk(1:125,1) = -1+(1+1).*rand(125,1);
% uk(1:125,2) = 0;
% % 90 and 270
% uk(126:250,2) = -1+(1+1).*rand(125,1);
% uk(126:250,1) = 0;
% % 45 and 225
% uk(251:375,1) = -1+(1+1).*rand(125,1);
% uk(251:375,2) = uk(251:375,1);
% % 135 and 325
% uk(376:500,1) = -1+(1+1).*rand(125,1);
% uk(376:500,2) = -1*uk(376:500,1);
% uk(1,1) = 1;uk(1,2) = 0;
% uk(2,1) = 0.7;uk(2,2) = 0.8;
% uk(3,1) = 0;uk(3,2) = 1;
% uk(4,1) = -0.7;uk(4,2) = 0.8;
% uk(5,1) = -1;uk(5,2) = 0;
% uk(6,1) = -0.7;uk(6,2) = -0.8;
% uk(7,1) = 0;uk(7,2) = -1;
% uk(8,1) = 0.7;uk(8,2) = -0.8;
% uk(:,1) = -1+(1+1).*rand(num_inputs,1);
% uk(:,2) = -1+(1+1).*rand(num_inputs,1);
% multi dir, few dist
% % one dir - 45 degree
% uk(:,1) = rand(num_inputs,1);
% uk(:,2) = uk(:,1);
% % two dir - 45 + 325 degree
% uk(1:num_inputs/2,1) = rand(num_inputs/2,1);
% uk(1:num_inputs/2,2) = uk(1:num_inputs/2,1);
% uk(num_inputs/2+1:num_inputs,1) = rand(num_inputs/2,1);
% uk(num_inputs/2+1:num_inputs,2) = -1*uk(num_inputs/2+1:num_inputs,1);
% % four dir - 45 + 135 + 225 +325 degree
% uk(1:num_inputs/2,1) = -1+(1+1).*rand(num_inputs/2,1);
% uk(1:num_inputs/2,2) = uk(1:num_inputs/2,1);
% uk(num_inputs/2+1:num_inputs,1) = -1+(1+1).*rand(num_inputs/2,1);
% uk(num_inputs/2+1:num_inputs,2) = -1*uk(num_inputs/2+1:num_inputs,1);
% eight dir - 0 + 45 + 90 + 135 + 180 + 225 + 270 +325 degree
% % 0 and 180
% uk(1:125,1) = -1+(1+1).*rand(125,1);
% uk(1:125,2) = 0;
% % 90 and 270
% uk(126:250,2) = -1+(1+1).*rand(125,1);
% uk(126:250,1) = 0;
% % 45 and 225
% uk(251:375,1) = -1+(1+1).*rand(125,1);
% uk(251:375,2) = uk(251:375,1);
% % 135 and 325
% uk(376:500,1) = -1+(1+1).*rand(125,1);
% uk(376:500,2) = -1*uk(376:500,1);
% uk(1,1) = 1;uk(1,2) = 0;
% uk(2,1) = 0.7;uk(2,2) = 0.8;
% uk(3,1) = 0;uk(3,2) = 1;
% uk(4,1) = -0.7;uk(4,2) = 0.8;
% uk(5,1) = -1;uk(5,2) = 0;
% uk(6,1) = -0.7;uk(6,2) = -0.8;
% uk(7,1) = 0;uk(7,2) = -1;
% uk(8,1) = 0.7;uk(8,2) = -0.8;
% uk(:,1) = -1+(1+1).*rand(num_inputs,1);
% uk(:,2) = -1+(1+1).*rand(num_inputs,1);
% restricted distances on two directions
% % 45 degree
uk(:,1) = rand(num_inputs,1);
uk(:,2) = -1*uk(:,1);
% uk(:,1) = 0.7+(0.3*rand(num_inputs,1));
% uk(:,2) = uk(:,1);
% 135 degree
% uk(:,2) = 0.3*rand(num_inputs,1);
% uk(:,1) = -1*uk(:,2);
% uk(:,2) = 0.7+(0.3*rand(num_inputs,1));
% uk(:,1) = -1*uk(:,2);
figure()
plot(uk(:,1),uk(:,2),'r*')
% % first quadrant diagonal
% uk(301:325,1) = -1+(1+1).*rand(25,1);
% uk(301:325,2) = uk(301:325,1);
% % second quadrant diagonal
% uk(326:350,2) = -1+(1+1).*rand(25,1);
% uk(326:350,1) = uk(326:350,2);
% % third quadrant diagonal
% uk(351:375,1) = -1.*rand(25,1);
% uk(351:375,2) = uk(351:375,1);
% % forth quadrant diagonal
% uk(376:400,2) = -1+(1+1).*rand(25,1);
% uk(376:400,1) = -1*uk(376:400,2);
% % uk(326:350,2) = -1+(1+1).*rand(25,1);
% % uk(326:350,1) = uk(326:350,2);
% uk(401:450,1) = -1+(1+1).*rand(50,1);
% uk(451:500,2) = -1+(1+1).*rand(50,1);
% % uk(497,:) = 1;uk(498,:) = -0.7;
% % uk(497,:) = -1;uk(498,:) = -0.8;
for i = 1:num_inputs
    ux = ones(20,1)*uk(i,1);
    uy = ones(20,1)*uk(i,2);
    wn = sqrt(1);
    x = zeros(20,1);
    y = zeros(20,1);
%     rot_x = zeros(20,1);
%     rot_y = zeros(20,1);
    for t = 1:time_steps
        x(t,1) = (1-(exp(-wn*t))-((wn*t)*exp(-wn*t)))*ux(t);
        y(t,1) = (1-(exp(-wn*t))-((wn*t)*exp(-wn*t)))*uy(t);
%         rot_x(t,1) = cosd(-15)*y(t,1)-sind(-15)*y(t,1);
%         rot_y(t,1) = sind(-15)*y(t,1)+cosd(-15)*y(t,1);
    end
%     resp_all(((i-1)*time_steps)+1:i*time_steps,3) = rot_x;
%     resp_all(((i-1)*time_steps)+1:i*time_steps,4) = rot_y;
    resp_all(((i-1)*time_steps)+1:i*time_steps,1) = uk(i,1);
    resp_all(((i-1)*time_steps)+1:i*time_steps,2) = uk(i,2);
    resp_all(((i-1)*time_steps)+2:i*time_steps,3) = x(1:19,1);
    resp_all(((i-1)*time_steps)+2:i*time_steps,4) = y(1:19,1);
    vel = zeros(time_steps-1,3);
    dist = zeros(time_steps,2);
    for tt = 1:time_steps-1
        vel(tt,1) = abs(x(tt,1)-x(tt+1,1));
        vel(tt,2) = abs(y(tt,1)-y(tt+1,1));
        CosTheta = max(min(dot([1,0],[ux(tt,1),uy(tt,1)])/(norm([1,0])*norm([ux(tt,1),uy(tt,1)])),1),-1);
        vel(tt,3) = real(acosd(CosTheta))/180;
        dist(tt,1) = (x(tt+1,1)-x(tt,1));%sqrt((x(tt+1,1)-x(tt,1))^2+(y(tt+1,1)-y(tt,1))^2);
        dist(tt,2) = (y(tt+1,1)-y(tt,1));
%         if dist(tt,1) < 0.0005
%             dist(tt,1) = 0;
%         end
    end
    resp_all(((i-1)*time_steps)+2:i*time_steps,5) = vel(:,1);
    resp_all(((i-1)*time_steps)+2:i*time_steps,6) = vel(:,2);
    resp_all(((i-1)*time_steps)+2:i*time_steps,7) = vel(:,3);
    resp_all(((i-1)*time_steps)+1:i*time_steps,8) = dist(:,1);
    resp_all(((i-1)*time_steps)+1:i*time_steps,9) = dist(:,2);
%     resp_all(i,:,1) = uk(i,1);
%     resp_all(i,:,2) = uk(i,2);
%     resp_all(i,2:20,3) = x(1:19,1);
%     resp_all(i,2:20,4) = y(1:19,1);
end
% data1 = zeros(num_inputs*40,4);
% data2 = zeros(num_inputs*40,4);
% for i = 1:num_inputs
%     b_1 = 1;
%     wn_1 = 1; % resonant frequency
%     sn_1 = 1; % damping coefficient
%     d_1 = 1; % feedback gain coefficient
%     sys_1 = tf(b_1*(wn_1^2),[1,2*sn_1*(wn_1^2),d_1*(wn_1^2)*b_1]);
%     step_1 = step(sys_1,t)*uk(i,1);
%     step_2 = step(sys_1,t)*uk(i,2);
%     d = zeros(40,4);
%     d(:,1) = uk(i,1);%vel((i-1)*5+1,:);
%     d(:,2) = uk(i,2);
%     for b = 1:20
%         d(5+b,3) = step_1((b-1)*5+1);
%         d(5+b,4) = step_2((b-1)*5+1);
%     end
%     d(26:40,3) = uk(i,1);
%     d(26:40,4) = uk(i,2);
%     for a = 1:4
%         data1(((i-1)*40)+1:i*40,a) = d(:,a);
%     end
%     ux = ones(time_steps,1)*uk(i,1);
%     uy = ones(time_steps,1)*uk(i,2);
%     wn = sqrt(1);
%     x = zeros(time_steps,1);
%     y = zeros(time_steps,1);
%     tt = 1;
%     for ttt = 0:0.1:19
%         x(tt,1) = (1-(exp(-wn*ttt))-((wn*ttt)*exp(-wn*ttt)))*ux(tt);
%         y(tt,1) = (1-(exp(-wn*ttt))-((wn*ttt)*exp(-wn*ttt)))*uy(tt);
%         tt = tt+1;   
%     end
%     d = zeros(40,4);
%     d(:,1) = uk(i,1);
%     d(:,2) = uk(i,2);
% %     d(1:5,3) = 1;
%     for c = 1:20
%         d(5+c,3) = x((c-1)*5+1);
%         d(5+c,4) = y((c-1)*5+1);
%     end
%     d(26:40,3) = uk(i,1);
%     d(26:40,4) = uk(i,2);
%     for e = 1:4
%         data2(((i-1)*40)+1:i*40,e) = d(:,e);
%     end
%     resp_all(((i-1)*time_steps)+1:i*time_steps,1) = uk(i,1);
%     resp_all(((i-1)*time_steps)+1:i*time_steps,2) = uk(i,2);
%     resp_all(((i-1)*time_steps)+1:i*time_steps,3) = x;
%     resp_all(((i-1)*time_steps)+1:i*time_steps,4) = y;
% end
% figure()
% plot(resp_all(1:500,1),'b')
% hold on
% plot(resp_all(1:500,3),'r')
% figure()
% plot(data1(1:40,1),'b')
% hold on
% plot(data1(1:40,3),'r*')
% figure()
% plot(data2(1:40,1),'b')
% hold on
% plot(data2(1:40,3),'r*')
% figure(2)
% plot(resp_all(:,3),resp_all(:,4),'b')
% hold on
% plot(resp_all(1:500,4),'r')
data = [0 0 0 0 0 0 0 0 0;resp_all];
xlswrite('data/train_315_degree.xlsx',data);
%%
% B = 1; J = 1; k = 1;
% T = 1;
% theta = zeros(100,1);
% for t = 3:100
%     theta(t) = (T + (theta(t-1)*(B+J))+ ((theta(t-2)*J)/2))/(k+B+(J/2));
% end
%%
% figure()
% for in = 1:5
%     vel = zeros(time_steps-1,3);
%     for i = 1:time_steps-1
%         vel(i,1) = abs(resp_all(in,i,3)-resp_all(in,i+1,3));
%         vel(i,2) = abs(resp_all(in,i,4)-resp_all(in,i+1,4));
%         vel(i,3) = sqrt((resp_all(in,i,3)-resp_all(in,i+1,3))^2+(resp_all(in,i,4)-resp_all(in,i+1,4))^2);
%     end
%     plot(vel(:,3),'r');
%     hold on
% end

% plot(vel(:,1),'b');
% hold on
% plot(vel(:,2),'g');
% hold on
% plot(vel(:,3),'r');
% legend('x','y','res_v');
% vel_d = zeros(40,3);
% for i = 1:20
%     vel_d(5+i,:) = vel((i-1)*5+1,:);
% end
% figure()
% plot(vel_d(:,1),'b');
% hold on
% plot(vel_d(:,2),'g');
% hold on
% plot(vel_d(:,3),'r');
% legend('x','y','res_v');
%% same distance
% clear all;close all; clc
% radius = 1;
% xCenter = 0;
% yCenter = 0;
% % theta = [0,45,90,135,180,225,270,315];
% theta = [0,22.5,45,67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5];
% %theta = linspace(0, 360, 8*pi*radius); % More than needed to avoid gaps.
% x = xCenter + radius * cosd(theta);
% y = yCenter + radius * sind(theta);
% figure(2)
% plot(x,y)



