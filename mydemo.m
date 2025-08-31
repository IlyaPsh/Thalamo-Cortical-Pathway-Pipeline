
clear all
close all

matDir  = 'C:\Users\gtec\OneDrive\Work\BachelorThesis\ssep_analysis_v2';
matFiles = {
    fullfile(matDir,'evokedNP20RecordSession_1_2025.06.24_10.35.00_even.mat')
    fullfile(matDir,'evokedNP30RecordSession_1_2025.06.24_10.35.00_odd.mat')
    fullfile(matDir,'evokedNP20RecordSession_2_2025.06.24_10.38.43_even.mat')
    fullfile(matDir,'evokedNP30RecordSession_2_2025.06.24_10.38.43_odd.mat')
};

matDir  = 'C:\Users\gtec\OneDrive\Work\BachelorThesis\ssep_analysis_average\RecordSession_1_2025.06.24_10.35.00';
matFiles = {
    fullfile(matDir,'evokedNP20_RecordSession_1_2025.06.24_10.35.00.mat')
    fullfile(matDir,'evokedNP30_RecordSession_1_2025.06.24_10.35.00.mat')
    %fullfile(matDir,'evokedNP20_RecordSession_2_2025.06.24_10.38.43.mat')
    %fullfile(matDir,'evokedNP30_RecordSession_2_2025.06.24_10.38.43.mat')
};

scaleUnit = @(x) (x - min(x(:))) ./ (max(x(:)) - min(x(:)));

for k = 1:numel(matFiles)
    fname = matFiles{k};
    fprintf('--- [%d/%d]  %s ---\n',k,numel(matFiles),fname);

    S = load(fname); 

    fNames = fieldnames(S);
    for i = 1:numel(fNames)
        fn = fNames{i};
        if startsWith(fn,'amp_')
            if isnumeric(S.(fn))
                S.(fn) = scaleUnit(S.(fn));
            end
        end
    end

    lefunction(S);
end

function lefunction(dat)
    channels = cellstr(dat.channel_names)';
    channels_char = char(channels(:,1));
    disp(channels)
    fnames    = fieldnames(dat);
    ampFields = fnames(startsWith(fnames,'amp_'));
    if isempty(ampFields)
        error('No amp_ fields found in the data struct.');
    end
    chosenAmp = ampFields{1};

    figure();
    mon = BrainModel(fullfile(cd, 'TestData', ...         
        'BrainModel_MNI.xml'));
    plot3D = MontagePlot3D(gca, mon, {'Head';'LhCortex';'RhCortex'}, channels);
    act = (dat.(chosenAmp)- 0.5) * 2; %(rand(8, 1) - 0.5) * 2;                        
    plot3D.interpolateOnSurface('Head', act);           
    plot3D.setColormap(parula(1000), [-1 1]);
    plot3D.showColorbar();
    h_head = plot3D.getSurface('Head');
    h_head.setOpacity(0.8);
    h_text = plot3D.plotText(dat.channel_names', 7);   
    h_text.setColor([1 1 1]);
    h_text.setBackgroundColor('none');
    radii = abs(act * 5);                                  
    h_sph=plot3D.plotBubbles(radii);

    nCh = numel(dat.channel_names);
    for i = 1:nCh
        if act(i) > 0
            h_sph.setElementColor(i, [1 0 0]);   % red for positive
        else
            h_sph.setElementColor(i, [0 0 1]);   % blue for non‑positive
        end
    end

    lighting gouraud;
    axis equal;
    
    title(sprintf('Showing %s', chosenAmp), 'Interpreter','none');
end
