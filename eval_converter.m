clear;
addpath('/media/nulledge/3rd/data/MPII/mpii_human_pose_v1_u12_2');
load 'mpii_human_pose_v1_u12_1';


annolist_test = RELEASE.annolist(RELEASE.img_train == 0);
single_person_test = RELEASE.single_person(RELEASE.img_train == 0);

n = 0;
annolist_flat = struct();

pred = annolist_test;

for img_idx = 1:length(annolist_test)
    rect = annolist_test(img_idx).annorect;
    for r_idx = 1:length(rect)
        if (isfield(rect(r_idx), 'objpos') ...
            && ~isempty(rect(r_idx).objpos) ...
            && ismember(r_idx, single_person_test{img_idx}))
            n = n + 1;
            
            annolist_flat(n).image.name = annolist_test(img_idx).image.name;
            annolist_flat(n).annorect = rect(r_idx);
            annolist_flat(n).img_idx = img_idx;
            annolist_flat(n).r_idx = r_idx; 
            
            dir = '/media/nulledge/2nd/ubuntu/bilinear/prediction/';
            file = strcat(dir, int2str(img_idx), '.', int2str(r_idx), '.txt');
            
            M = dlmread(file);
            
            for joint_idx = 1:16
                x = M(joint_idx, 2);
                y = M(joint_idx, 3);
                
                pred(img_idx).annorect(r_idx).annopoints.point(joint_idx).id = joint_idx - 1;
                pred(img_idx).annorect(r_idx).annopoints.point(joint_idx).x = x;
                pred(img_idx).annorect(r_idx).annopoints.point(joint_idx).y = y;
                
            end
            clear x y;
        end
        clear dir file M;
    end
    clear r_idx rect;
end
clear img_idx;

save('pred_keypoints_mpii.mat', 'pred');