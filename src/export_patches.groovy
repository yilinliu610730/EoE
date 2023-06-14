import qupath.lib.images.servers.LabeledImageServer

def project = getProject()
for (entry in project.getImageList()) {
    def imageData = entry.readImageData()
    // Define output path (relative to project)
    def name = entry.getImageName()
    
    def pathOutput = buildFilePath('YourDirectory', name_post.toString())
    mkdirs(pathOutput)
    
    print(pathOutput)
        
    // Convert to downsample
    double downsample = 1
    
    // Create an exporter that requests corresponding tiles from the original & labeled image servers
    new TileExporter(imageData)
        .includePartialTiles(false)
        .downsample(1)              // Define export resolution
        .imageExtension('.jpg')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
        .tileSize(512)              // Define size of each tile, in pixels
        .annotatedTilesOnly(true)   // If true, only export tiles if there is a (labeled) annotation present
        .overlap(256)               // Define overlap, in pixel units at the export resolution
        .writeTiles(pathOutput)     // Write tiles to the specified directory
    
    print 'Done!'
}

