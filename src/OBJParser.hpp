#ifndef OBJ_PARSER
#define OBJ_PARSER

#include <Eigen/Dense>

#include <string>
#include <sstream>
#include <assert.h>
#include <fstream>

// Very basic parser for .obj files, to handle [x,y,z] vertices and faces without texture
class OBJParser {
    public:
        OBJParser(std::string filename) {
            obj_ifs = new std::ifstream(filename.c_str());
            if(!(*obj_ifs).good()) {
                std::cout << "Error opening OBJ file!\n";
            }
            fileRead = false; // file has not yet been read
            meshGenerated = false;
        }

        bool readFile() {
            verts.clear();
            faces.clear();
            std::string line;
            while(std::getline(*obj_ifs, line)) {
                double x, y, z;
                if (3 == sscanf(line.c_str(), "v %lf, %lf, %lf", &x, &y, &z)) {
                    Eigen::Vector3d pt = Eigen::Vector3d(x, y, z);
                    verts.push_back(pt);
                } else {
                    if(line.at(0) == 'f') { // list of vertices constituting the face to follow
                        std::string cur_face;
                        std::vector<int> v_is;
                        std::stringstream vl_stream(line.substr(2));
                        while(std::getline(vl_stream, cur_face, ' ')) {
                            int cur_i = atoi(cur_face.c_str());
                            v_is.push_back(cur_i);
                        }
                        faces.push_back(v_is);
                    }
                }
            }
            (*obj_ifs).close();
            fileRead = true;
            return true;
        }

        std::vector<Eigen::Vector3d> getVerts() {
            if(!fileRead)
                std::cout << "Error! OBJ file not yet read.\n";
            return verts;
        }

        std::vector< std::vector<int> > getFaces() {
            if(!fileRead)
                std::cout << "Error! OBJ file not yet read.\n";
            return faces;
        }

        bool generateMeshEdges() {
            if(!fileRead) return false;
            if(meshGenerated) return true;
            for(unsigned int fi = 0; fi < faces.size(); fi++) {
                std::vector<int> v_list = faces[fi];
                for(unsigned int vi = 0; vi < v_list.size(); vi++) {
                    // Get all edges of the face 0->1, 1->2 ... n->0
                    int x = v_list[vi];
                    int y = v_list[(vi + 1) % v_list.size()];
                    assert(x < (int)verts.size() && y < (int)verts.size());
                    std::pair<int, int> p = std::pair<int,int>(x,y);
                    edges.push_back(p);
                }
            }
            meshGenerated = true;
            return true;
        }

        std::vector< std::pair<int, int> > getEdges() {
            if(!fileRead)
                std::cout << "Error! OBJ file not yet read.\n";
            return edges;
        }

    private:
        bool fileRead, meshGenerated;
        std::ifstream *obj_ifs;
        std::vector<Eigen::Vector3d> verts;
        // Edges is a vector of pairs, describing indeces of adjacent vertices  
        std::vector< std::pair<int, int> > edges;
        std::vector< std::vector<int> > faces;
};
#endif
