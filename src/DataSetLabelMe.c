#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/DataSetLabelMe.c"
#else

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

static int nn_(DataSetLabelMe_extract)(lua_State *L)
{
  // fprintf(stderr,"test!");
  int tags = 1;
  THTensor *mask = luaT_checkudata(L, 2, torch_Tensor);
  int x_start = lua_tonumber(L, 3);
  int x_end = lua_tonumber(L, 4);
  int y_start = lua_tonumber(L, 5);
  int y_end = lua_tonumber(L, 6);
  int idx = lua_tonumber(L, 7);
  float filter_ratio = lua_tonumber(L, 8);
  int filter_size = lua_tonumber(L, 9);
  int filter_step = lua_tonumber(L, 10);
//fprintf(stderr,"test2!");
  float ratio = 1;
  int x,y,label,tag,size;
  THShortStorage *data;
  for (x=x_start; x<=x_end; x++) {

    for (y=y_start; y<=y_end; y++) {
      //fprintf(stderr,"-");
      // label = mask[x][y]
      label = THTensor_(get2d)(mask, y-1, x-1);

      // optional filter: insures that at least N% of local pixels belong to the same class
      if (filter_ratio > 0) {
        int kx,ky,count=0,good=0;
        for (kx=MAX(1,x-filter_size/2); kx<=MIN(x_end,x+filter_size/2); kx+=filter_step) {
          for (ky=MAX(1,y-filter_size/2); ky<=MIN(y_end,y+filter_size/2); ky+=filter_step) {
            int other = THTensor_(get2d)(mask, ky-1, kx-1);
            if (other == label) good++;
            count++;
          }
        }
        ratio = (float)good/(float)count;
      }

      // if filter(s) satisfied, then append label
      if (ratio >= filter_ratio) {

        lua_rawgeti(L, tags, label);                                              // tag = tags[label]
        tag = lua_gettop(L);

        lua_pushstring(L, "size"); lua_rawget(L, tag);                            // size = tag.size
        size = lua_tonumber(L,-1); lua_pop(L,1);

        lua_pushstring(L, "size"); lua_pushnumber(L, size+3); lua_rawset(L, tag); // tag.size = size + 3

        lua_pushstring(L, "data"); lua_rawget(L, tag);                            // data = tag.data
        data = luaT_checkudata(L, -1, "torch.ShortStorage"); lua_pop(L, 1);
        //fprintf(stderr,"t4");
        data->data[size] = x;                                                     // data[size+1] = x
        data->data[size+1] = y;                                                   // data[size+1] = y
        data->data[size+2] = idx;                                                 // data[size+1] = idx

        lua_pop(L, 1);
      }
    }
  }
  return 0;
}

/******************************************************/
// Camille : same function that below except it keeps memory about the employed masking segment.

static int nn_(DataSetSegmentSampling_extract)(lua_State *L)
{
  int tags = 1;
  THTensor *mask = luaT_checkudata(L, 2, torch_Tensor);
  int x_start = lua_tonumber(L, 3);
  int x_end = lua_tonumber(L, 4);
  int y_start = lua_tonumber(L, 5);
  int y_end = lua_tonumber(L, 6);
  int idx = lua_tonumber(L, 7);
 int idxSegm = lua_tonumber(L, 8);
  float filter_ratio = lua_tonumber(L, 9);
  int filter_size = lua_tonumber(L, 10);
  int filter_step = lua_tonumber(L, 11);
  int step = lua_tonumber(L, 12);
  float ratio = 1;
  int x,y,label,tag,size;
  THShortStorage *data;
  for (x=x_start; x<=x_end; x=x+step) {
    for (y=y_start; y<=y_end; y=y+step) {
      // label = mask[x][y]
      label = THTensor_(get2d)(mask, y-1, x-1);
      //  fprintf(stderr,"%d %d \n",x,y);
      // optional filter: insures that at least N% of local pixels belong to the same class
      if (filter_ratio > 0) {
        int kx,ky,count=0,good=0;
        for (kx=MAX(1,x-filter_size/2); kx<=MIN(x_end,x+filter_size/2); kx+=filter_step) {
          for (ky=MAX(1,y-filter_size/2); ky<=MIN(y_end,y+filter_size/2); ky+=filter_step) {
            int other = THTensor_(get2d)(mask, ky-1, kx-1);
            if (other == label) good++;
            count++;
          }
        }
        ratio = (float)good/(float)count;
      }

      // if filter(s) satisfied, then append label
      if (ratio >= filter_ratio) {
        lua_rawgeti(L, tags, label);                                              // tag = tags[label]
        tag = lua_gettop(L);
        lua_pushstring(L, "size"); lua_rawget(L, tag);                            // size = tag.size
        size = lua_tonumber(L,-1); lua_pop(L,1);
        lua_pushstring(L, "size"); lua_pushnumber(L, size+4); lua_rawset(L, tag); // tag.size = size + 4
        lua_pushstring(L, "data"); lua_rawget(L, tag);                            // data = tag.data
        data = luaT_checkudata(L, -1, "torch.ShortStorage"); lua_pop(L, 1);
        data->data[size] = x;                                                     // data[size+1] = x
        data->data[size+1] = y;                                                   // data[size+1] = y
        data->data[size+2] = idx;                                                 // data[size+1] = idx
        data->data[size+3] = idxSegm;                                                 // data[size+1] = idxSegm
        lua_pop(L, 1);
      }
    }
  }
  return 0;
}


static const struct luaL_Reg nn_(DataSetLabelMe__) [] = {
  {"DataSetLabelMe_extract", nn_(DataSetLabelMe_extract)},
  {"DataSetSegmentSampling_extract", nn_(DataSetSegmentSampling_extract)},
  {NULL, NULL}
};



static void nn_(DataSetLabelMe_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(DataSetLabelMe__), "nn");
  lua_pop(L,1);
}

#endif
