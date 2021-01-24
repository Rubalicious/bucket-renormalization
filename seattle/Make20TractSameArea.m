clc
clear
numRetainedTracts=20;
travelData=table2array(readtable("seattle_travel_numbers.csv"));
gisData=readtable("seattle_GIS_data.csv");
uVData=table2array(readtable("seattle_UV_coordinates.csv"));
tenTractIndices=table2array(readtable("seattle_10_sameArea_indices.csv"));
twentyTractIndices=table2array(readtable("seattle_20_sameArea_indices.csv"));
travelData10(10,10)=0;
travelData20(20,20)=0;
groupIndices10=cell(10,1);
for i=1:size(travelData,1)
    newIndex=tenTractIndices(i,1);
    groupIndices10{newIndex,1}(size(groupIndices10{newIndex,1},2)+1)=tenTractIndices(i,2);
end
for i=1:size(groupIndices10,1)
    for j=1:size(groupIndices10{i,1},2)
        for k=1:size(groupIndices10,1)
            if i~=k
                for m=1:size(groupIndices10{k,1},2)
                    travelData10(i,k)=travelData10(i,k)+travelData(j,m);
                end
            end
        end
    end
end
output_gisData_10=gisData;
output_gisData_10(11:size(gisData,1),:)=[];
for i=1:10
    avgLat=0;
    avgLon=0;
    avgPopulatio=0;
    avgDensity=0;
    for j=1:size(groupIndices10{i,1},2)
        avgLat=avgLat+table2array(gisData(groupIndices10{i,1}(1,j)+1,2));
        avgLon=avgLon+table2array(gisData(groupIndices10{i,1}(1,j)+1,3));
        avgPopulatio=avgPopulatio+table2array(gisData(groupIndices10{i,1}(1,j)+1,4));
        avgDensity=avgDensity+table2array(gisData(groupIndices10{i,1}(1,j)+1,5));
    end
    avgLat=avgLat/size(groupIndices10{i,1},2);
    avgLon=avgLon/size(groupIndices10{i,1},2);
    avgPopulatio=avgPopulatio/size(groupIndices10{i,1},2);
    avgDensity=avgDensity/size(groupIndices10{i,1},2);
    output_gisData_10(i,2)=array2table(avgLat);
    output_gisData_10(i,3)=array2table(avgLon);
    output_gisData_10(i,4)=array2table(avgPopulatio);
    output_gisData_10(i,5)=array2table(avgDensity);
end
uVData10(10,3)=0;
for i=1:10
    avgU=0;
    avgV=0;
    for j=1:size(groupIndices10{i,1},2)
        avgU=avgU+uVData(groupIndices10{i,1}(1,j)+1,2);
        avgV=avgV+uVData(groupIndices10{i,1}(1,j)+1,3);
    end
    avgU=avgU/size(groupIndices10{i,1},2);
    avgV=avgV/size(groupIndices10{i,1},2);
    uVData10(i,2)=avgU;
    uVData10(i,3)=avgV;
    uVData10(i,1)=i;
end
writetable(output_gisData_10,strcat('seattle_10sameArea_GIS_data.csv'));
writetable(array2table(uVData10),strcat('seattle_10sameArea_UV_coordinates.csv'),'WriteVariableNames',false);
writetable(array2table(travelData10),strcat('seattle_10sameArea_travel_numbers.csv'),'WriteVariableNames',false);

groupIndices20=cell(20,1);
for i=1:size(travelData,1)
    newIndex=twentyTractIndices(i,1);
    groupIndices20{newIndex,1}(size(groupIndices20{newIndex,1},2)+1)=twentyTractIndices(i,2);
end
for i=1:size(groupIndices20,1)
    for j=1:size(groupIndices20{i,1},2)
        for k=1:size(groupIndices20,1)
            if i~=k
                for m=1:size(groupIndices20{k,1},2)
                    travelData20(i,k)=travelData20(i,k)+travelData(j,m);
                end
            end
        end
    end
end
output_gisData_20=gisData;
output_gisData_20(21:size(gisData,1),:)=[];
for i=1:20
    avgLat=0;
    avgLon=0;
    avgPopulatio=0;
    avgDensity=0;
    for j=1:size(groupIndices20{i,1},2)
        avgLat=avgLat+table2array(gisData(groupIndices20{i,1}(1,j)+1,2));
        avgLon=avgLon+table2array(gisData(groupIndices20{i,1}(1,j)+1,3));
        avgPopulatio=avgPopulatio+table2array(gisData(groupIndices20{i,1}(1,j)+1,4));
        avgDensity=avgDensity+table2array(gisData(groupIndices20{i,1}(1,j)+1,5));
    end
    avgLat=avgLat/size(groupIndices20{i,1},2);
    avgLon=avgLon/size(groupIndices20{i,1},2);
    avgPopulatio=avgPopulatio/size(groupIndices20{i,1},2);
    avgDensity=avgDensity/size(groupIndices20{i,1},2);
    output_gisData_20(i,2)=array2table(avgLat);
    output_gisData_20(i,3)=array2table(avgLon);
    output_gisData_20(i,4)=array2table(avgPopulatio);
    output_gisData_20(i,5)=array2table(avgDensity);
end
uVData20(20,3)=0;
for i=1:20
    avgU=0;
    avgV=0;
    for j=1:size(groupIndices20{i,1},2)
        avgU=avgU+uVData(groupIndices20{i,1}(1,j)+1,2);
        avgV=avgV+uVData(groupIndices20{i,1}(1,j)+1,3);
    end
    avgU=avgU/size(groupIndices20{i,1},2);
    avgV=avgV/size(groupIndices20{i,1},2);
    uVData20(i,2)=avgU;
    uVData20(i,3)=avgV;
    uVData20(i,1)=i;
end
writetable(output_gisData_20,strcat('seattle_20sameArea_GIS_data.csv'));
writetable(array2table(uVData20),strcat('seattle_20sameArea_UV_coordinates.csv'),'WriteVariableNames',false);
writetable(array2table(travelData20),strcat('seattle_20sameArea_travel_numbers.csv'),'WriteVariableNames',false);