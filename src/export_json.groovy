import qupath.lib.io.GsonTools
import qupath.lib.images.servers.LabeledImageServer

def project = getProject()
for (entry in project.getImageList()) {
    def imageData = entry.readImageData()
    def hierarchy = imageData.getHierarchy()
    def annotations = hierarchy.getAnnotationObjects()
    def name = entry.getImageName()

    def OUTPUT_DIR = 'Your Directory' 
    //def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
    def filePath = buildFilePath(OUTPUT_DIR, name.toString())
    mkdirs(filePath)
    boolean prettyPrint = true
    def gson = GsonTools.getInstance(prettyPrint)
    def writer = new FileWriter(filePath);
    gson.toJson(annotations,writer)
    writer.flush()
    print("done")
}

