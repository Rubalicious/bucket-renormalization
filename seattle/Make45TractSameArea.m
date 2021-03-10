clc
clear
numRetainedTracts=45;
travelData=table2array(readtable("seattle_travel_numbers.csv"));
gisData=readtable("seattle_GIS_data.csv");
uVData=table2array(readtable("seattle_UV_coordinates.csv"));
fortyFiveTractIndices=table2array(readtable("seattle_45_sameArea_indices.csv"));
travelData45(numRetainedTracts,numRetainedTracts)=0;
groupIndices45=cell(numRetainedTracts,1);
for i=1:size(travelData,1)
    newIndex=fortyFiveTractIndices(i,1);
    groupIndices45{newIndex,1}(size(groupIndices45{newIndex,1},2)+1)=fortyFiveTractIndices(i,2);
end
for i=1:size(groupIndices45,1)
    for j=1:size(groupIndices45{i,1},2)
        for k=1:size(groupIndices45,1)
            if i~=k
                for m=1:size(groupIndices45{k,1},2)
                    travelData45(i,k)=travelData45(i,k)+travelData(groupIndices45{i,1}(1,j)+1,groupIndices45{k,1}(1,m)+1);
                end
            end
        end
    end
end
output_gisData_45=gisData;
output_gisData_45(numRetainedTracts+1:size(gisData,1),:)=[];
for i=1:numRetainedTracts
    avgLat=0;
    avgLon=0;
    avgPopulatio=0;
    avgDensity=0;
    for j=1:size(groupIndices45{i,1},2)
        avgLat=avgLat+table2array(gisData(groupIndices45{i,1}(1,j)+1,2));
        avgLon=avgLon+table2array(gisData(groupIndices45{i,1}(1,j)+1,3));
        avgPopulatio=avgPopulatio+table2array(gisData(groupIndices45{i,1}(1,j)+1,4));
        avgDensity=avgDensity+table2array(gisData(groupIndices45{i,1}(1,j)+1,5));
    end
    avgLat=avgLat/size(groupIndices45{i,1},2);
    avgLon=avgLon/size(groupIndices45{i,1},2);
    avgPopulatio=avgPopulatio/size(groupIndices45{i,1},2);
    avgDensity=avgDensity/size(groupIndices45{i,1},2);
    output_gisData_45(i,2)=array2table(avgLat);
    output_gisData_45(i,3)=array2table(avgLon);
    output_gisData_45(i,4)=array2table(avgPopulatio);
    output_gisData_45(i,5)=array2table(avgDensity);
end
uVData45(numRetainedTracts,3)=0;
for i=1:numRetainedTracts
    avgU=0;
    avgV=0;
    for j=1:size(groupIndices45{i,1},2)
        avgU=avgU+uVData(groupIndices45{i,1}(1,j)+1,2);
        avgV=avgV+uVData(groupIndices45{i,1}(1,j)+1,3);
    end
    avgU=avgU/size(groupIndices45{i,1},2);
    avgV=avgV/size(groupIndices45{i,1},2);
    uVData45(i,2)=avgU;
    uVData45(i,3)=avgV;
    uVData45(i,1)=i;
end
writetable(output_gisData_45,strcat('seattle_45sameArea_GIS_data.csv'));
writetable(array2table(uVData45),strcat('seattle_45sameArea_UV_coordinates.csv'),'WriteVariableNames',false);
writetable(array2table(travelData45),strcat('seattle_45sameArea_travel_numbers.csv'),'WriteVariableNames',false);