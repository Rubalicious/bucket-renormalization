clc
clear
numRetainedTracts=70;
travelData=table2array(readtable("seattle_travel_numbers.csv"));
gisData=readtable("seattle_GIS_data.csv");
uVData=table2array(readtable("seattle_UV_coordinates.csv"));
seventyTractIndices=table2array(readtable("seattle_70_sameArea_indices.csv"));
travelData70(numRetainedTracts,numRetainedTracts)=0;
groupIndices70=cell(numRetainedTracts,1);
for i=1:size(travelData,1)
    newIndex=seventyTractIndices(i,1);
    groupIndices70{newIndex,1}(size(groupIndices70{newIndex,1},2)+1)=seventyTractIndices(i,2);
end
for i=1:size(groupIndices70,1)
    for j=1:size(groupIndices70{i,1},2)
        for k=1:size(groupIndices70,1)
            if i~=k
                for m=1:size(groupIndices70{k,1},2)
                    travelData70(i,k)=travelData70(i,k)+travelData(groupIndices70{i,1}(1,j)+1,groupIndices70{k,1}(1,m)+1);
                end
            end
        end
    end
end
output_gisData_70=gisData;
output_gisData_70(numRetainedTracts+1:size(gisData,1),:)=[];
for i=1:numRetainedTracts
    avgLat=0;
    avgLon=0;
    avgPopulatio=0;
    avgDensity=0;
    for j=1:size(groupIndices70{i,1},2)
        avgLat=avgLat+table2array(gisData(groupIndices70{i,1}(1,j)+1,2));
        avgLon=avgLon+table2array(gisData(groupIndices70{i,1}(1,j)+1,3));
        avgPopulatio=avgPopulatio+table2array(gisData(groupIndices70{i,1}(1,j)+1,4));
        avgDensity=avgDensity+table2array(gisData(groupIndices70{i,1}(1,j)+1,5));
    end
    avgLat=avgLat/size(groupIndices70{i,1},2);
    avgLon=avgLon/size(groupIndices70{i,1},2);
    avgPopulatio=avgPopulatio/size(groupIndices70{i,1},2);
    avgDensity=avgDensity/size(groupIndices70{i,1},2);
    output_gisData_70(i,2)=array2table(avgLat);
    output_gisData_70(i,3)=array2table(avgLon);
    output_gisData_70(i,4)=array2table(avgPopulatio);
    output_gisData_70(i,5)=array2table(avgDensity);
end
uVData70(numRetainedTracts,3)=0;
for i=1:numRetainedTracts
    avgU=0;
    avgV=0;
    for j=1:size(groupIndices70{i,1},2)
        avgU=avgU+uVData(groupIndices70{i,1}(1,j)+1,2);
        avgV=avgV+uVData(groupIndices70{i,1}(1,j)+1,3);
    end
    avgU=avgU/size(groupIndices70{i,1},2);
    avgV=avgV/size(groupIndices70{i,1},2);
    uVData70(i,2)=avgU;
    uVData70(i,3)=avgV;
    uVData70(i,1)=i;
end
writetable(output_gisData_70,strcat('seattle_70sameArea_GIS_data.csv'));
writetable(array2table(uVData70),strcat('seattle_70sameArea_UV_coordinates.csv'),'WriteVariableNames',false);
writetable(array2table(travelData70),strcat('seattle_70sameArea_travel_numbers.csv'),'WriteVariableNames',false);