import os, json
import pathlib
from pathlib import Path
from pandas import DataFrame
import math

import honeybee_ies as hb2ies
from honeybee.model import Model as HBModel
from ladybug_geometry.geometry2d.pointvector import Vector2D
from honeybee_vtk.model import Model as VTKModel

from pollination_streamlit_viewer import viewer
from pollination_streamlit_io import get_hbjson


import streamlit as st

st.set_page_config(page_title='SpaceXtract',
    layout="wide"
)

#Side BAR
with st.sidebar:
    st.image('./img/diagram.png',use_column_width='auto',output_format='PNG',
             caption='This tool accepts .json files exported from Ladybyug-tools and will automatically extracts building envelope information. You can also export the 3D model into a .gem file for IES users. More features are to be developed!')

    st.header("Control Panel:")

    north_ = st.number_input("North Angle (90° by default):", -180,180, 90 , 1, key = 'north',help = "Counter-Clockwise for Negative Values, Clockwise for Positive Values")
    vectors = [math.cos(math.radians(north_)), math.sin(math.radians(north_))]
    

#MAIN PAGE

st.title("SpaceXtract")
st.subheader("Before uploading, please check the following items:")
st.markdown("**1- Each room should have unique names to calculate the internal wall surface areas accurately!**")
st.markdown("**2- Boundary conditions of adjacent surfaces should be done prior to use the tool, otherwise external wall surface areas will include internal walls!**")
st.markdown("**3- If you are interested to calculate the conditioned spaces only, they should be set as conditioned in the model!**")



st.subheader("Upload .hbjson Model:")


if 'temp' not in st.session_state:
     st.session_state.temp = Path('/tmp')
     st.session_state.temp.mkdir(parents=True, exist_ok=True)

def create_vtkjs(hbjson_path: Path):
    if not hbjson_path:
        return
    
    model = VTKModel.from_hbjson(hbjson_path.as_posix())
    
    vtkjs_dir = st.session_state.temp.joinpath('vtkjs')
    if not vtkjs_dir.exists():
        vtkjs_dir.mkdir(parents=True, exist_ok=True)

    vtkjs_file = vtkjs_dir.joinpath(f'{hbjson_path.stem}.vtkjs')
    
    if not vtkjs_file.is_file():
        model.to_vtkjs(
            folder=vtkjs_dir.as_posix(),
            name=hbjson_path.stem
        )

    return vtkjs_file

def show_model(hbjson_path: Path):
    """Render HBJSON."""

    vtkjs_name = f'{hbjson_path.stem}_vtkjs'
    vtkjs = create_vtkjs(hbjson_path)
    st.session_state.content = vtkjs.read_bytes()
    st.session_state[vtkjs_name] = vtkjs
    
    
def callback_once():
    if 'hbjson' in st.session_state.get_hbjson:
        hb_model = HBModel.from_dict(st.session_state.get_hbjson['hbjson'])
        if hb_model:
            hbjson_path = st.session_state.temp.joinpath(f'{hb_model.identifier}.hbjson')
            hbjson_path.write_text(json.dumps(hb_model.to_dict()))
            show_model(hbjson_path)
    return hb_model


hbjson = get_hbjson('get_hbjson', on_change=callback_once)

if st.session_state.get_hbjson is not None:
    st.subheader(f'Visualizing {callback_once().display_name} Model')
    if 'content' in st.session_state:
        viewer(
            content=st.session_state.content,
            key='vtkjs-viewer',
            subscribe=False,
            style={
                'height' : '640px'
            }
        )


else:
    st.info('Load a model!')

# EXPORT AS GEM FILE
path_to_out_folder = pathlib.Path('./gem')
path_to_out_folder.mkdir(parents=True, exist_ok=True)

if st.button("**Export as .GEM File (for IES Users)**"):
        
    data=hb2ies.writer.model_to_ies(model =callback_once(), folder = path_to_out_folder)


#Generating the model

with st.sidebar:
    area_calc_method = st.radio("Select the Facade Area Calculation Methodology", options = ['Conditioned Zones', 'Entire Building'])

def model_info() -> DataFrame:

    model = callback_once()

    #Extracting properties based on rooms
    model_data = {'Conditioning Status':[], 'Program Type': [], 'volume (m3)': [],'floor_area (m2)': [],'roof_area (m2)':[],  'exterior_wall_area (m2)': [],
                  'exterior_aperture_area (m2)': [], 'exterior_skylight_area (m2)':[],'display_name':[]}

    internal_srf_area = []
    room_id = []
    for room in model.rooms:
        model_data['display_name'].append(room.display_name)
        model_data['Conditioning Status'].append(room.properties.energy.is_conditioned)
        model_data['Program Type'].append(room.properties.energy.program_type._identifier)
        model_data['volume (m3)'].append(room.volume)
        model_data['floor_area (m2)'].append(room.floor_area)
        model_data['roof_area (m2)'].append(room.exterior_roof_area)
        model_data['exterior_wall_area (m2)'].append(room.exterior_wall_area - room.exterior_aperture_area)
        model_data['exterior_aperture_area (m2)'].append(room.exterior_aperture_area)
        model_data['exterior_skylight_area (m2)'].append(room.exterior_skylight_aperture_area)
        #internal walls   
        for face in room.faces:
            room_id.append(room.display_name)
            if face.boundary_condition.name == 'Surface':
                internal_srf_area.append(face.area)
            else:
                internal_srf_area.append(0)
        internal = DataFrame([room_id,internal_srf_area],['ROOM_ID','Internal_Faces']).transpose().sort_values('ROOM_ID').groupby('ROOM_ID').sum()

    model_data = DataFrame.from_dict(model_data).sort_values('display_name').set_index('display_name')

    model_data['internal_wall_area (m2)'] = internal['Internal_Faces']

    model_shade = {'total_external_shades_area (m2)':[]}

    for shade in model.outdoor_shades:
        model_shade['total_external_shades_area (m2)'].append(shade.area)
    
    return model_data, DataFrame.from_dict(model_shade).sum()


#Extracting room index based on facade calc methodology 

def facade_calc() -> DataFrame:

    model = callback_once()

    target_rooms_index = []

    for i,room in enumerate(range(len(model_info()[0].index))):
        
        if (area_calc_method == 'Conditioned Zones') and (model_info()[0]['Conditioning Status'].iloc[room] == True):
            target_rooms_index.append(i)
            
        elif area_calc_method == 'Entire Building':
            target_rooms_index.append(i)

    #Apertures based on directions
    aperture_azimuth = []
    aperture_area = []
    verti_faces_azimuth = []
    verti_faces_area = []
    roof_faces_area = []
    floor_faces_area = []
    

    model_apertures = {'Aperture Orientation':[],'Aperture Area (m2)':[]}
    model_faces_vertical = {'Wall Orientation':[],'Wall Area (m2)':[]}
    model_roof = {'Roof Area (m2)':[]}
    model_floor = {'Floor Area (m2)':[]}

    for room_index in target_rooms_index:
        for aperture in range(len(model.rooms[room_index].exterior_apertures)):

            if model.rooms[room_index].exterior_apertures[aperture].azimuth > 0: #excluding skylights if any
                
                aper_azimuth = model.rooms[room_index].exterior_apertures[aperture].horizontal_orientation(north_vector=Vector2D(vectors[0],vectors[1]))
                
                if (aper_azimuth <= 45) or (aper_azimuth > 315):
                    aperture_orientation = 'North'
                    aperture_ar = model.rooms[room_index].exterior_apertures[aperture].area
                elif (aper_azimuth > 45) and (aper_azimuth <= 135):
                    aperture_orientation = 'East'
                    aperture_ar = model.rooms[room_index].exterior_apertures[aperture].area
                elif (aper_azimuth > 135) and (aper_azimuth <= 225):
                    aperture_orientation = 'South'
                    aperture_ar = model.rooms[room_index].exterior_apertures[aperture].area
                elif (aper_azimuth > 225) and (aper_azimuth <= 315):
                    aperture_orientation = 'West'
                    aperture_ar = model.rooms[room_index].exterior_apertures[aperture].area
                
                aperture_azimuth.append(aperture_orientation)
                aperture_area.append(round(aperture_ar,2))

                model_apertures['Aperture Orientation'] = aperture_azimuth
                model_apertures['Aperture Area (m2)'] = aperture_area
        
    for room_index in target_rooms_index:
        for face in range(len(model.rooms[room_index].faces)):
            #Filtering external faces only
            if (model.rooms[room_index].faces[face].boundary_condition.name == 'Outdoors') and (model.rooms[room_index].faces[face].type.name != 'RoofCeiling') and (model.rooms[room_index].faces[face].type.name != 'Floor'):
                #checking vertical faces
                    
                vert_face_azimuth = model.rooms[room_index].faces[face].horizontal_orientation(north_vector=Vector2D(vectors[0],vectors[1]))
                # vert_face_area = model.rooms[room_index].faces[face].area
                if vert_face_azimuth <= 45 or vert_face_azimuth > 315:
                    face_orientation = 'North'
                    vert_face_area = model.rooms[room_index].faces[face].area
                elif vert_face_azimuth > 45 and vert_face_azimuth <= 135:
                    face_orientation = 'East'
                    vert_face_area = model.rooms[room_index].faces[face].area
                elif vert_face_azimuth > 135 and vert_face_azimuth <= 225:
                    face_orientation = 'South'
                    vert_face_area = model.rooms[room_index].faces[face].area
                elif vert_face_azimuth > 225 and vert_face_azimuth <= 315:
                    face_orientation = 'West'
                    vert_face_area = model.rooms[room_index].faces[face].area
                
                verti_faces_azimuth.append(face_orientation)
                verti_faces_area.append(round(vert_face_area,2))

                model_faces_vertical['Wall Orientation'] = verti_faces_azimuth
                model_faces_vertical['Wall Area (m2)'] = verti_faces_area
        
            #checking horizontal faces
            if model.rooms[room_index].faces[face].type.name == 'RoofCeiling':
                horiz_face_area = model.rooms[room_index].faces[face].area
                roof_faces_area.append(round(horiz_face_area,2))
                model_roof['Roof Area (m2)'] = roof_faces_area
            if model.rooms[room_index].faces[face].type.name == 'Floor': #if there is any exposed floors, if not returns 0
                horiz_face_area = model.rooms[room_index].faces[face].area
                floor_faces_area.append(round(horiz_face_area,2))
                model_floor['Floor Area (m2)'] = floor_faces_area

    return target_rooms_index, DataFrame.from_dict(model_apertures).groupby('Aperture Orientation').sum(),DataFrame.from_dict(model_faces_vertical).groupby('Wall Orientation').sum(),DataFrame.from_dict(model_roof).sum(), DataFrame.from_dict(model_floor).sum()


if st.session_state.get_hbjson is not None:
    #Building Relative Compactness (RC) = 6 * Building Volume (V) ^ 2/3 / Building Surface Area (A)
    ##Source: https://www.sciencedirect.com/science/article/abs/pii/S037877881400574X?via%3Dihub

    building_volume = model_info()[0]['volume (m3)'].sum()
    Building_area = model_info()[0][['floor_area (m2)','roof_area (m2)','exterior_wall_area (m2)','exterior_aperture_area (m2)','exterior_skylight_area (m2)']].sum().sum() #internal walls excluded 

    build_RC = (6 * (pow(building_volume,2/3))) / Building_area

    st.header(f'**Building Relative Compactness (RC)** is :red[{round(build_RC,2)}].')

    st.markdown('---')

    st.subheader(f'**Building General Details**')
    st.dataframe(model_info()[0], use_container_width=True)
    st.markdown('---')
    
    st.subheader(f'**Thermal Envelope Area Calculation Based on {area_calc_method}**')
    
    if facade_calc()[0] == []:
        st.subheader(":red[OOPS! NO CONDITIONED ZONES ARE ASSIGENED IN THE MODEL!]")
    else:
        cols = st.columns(4)

        with cols[0]:
            st.dataframe(model_info()[1], use_container_width=True)
            
            col_wwr = st.columns(len(facade_calc()[1].index))

            for metric in range(len(facade_calc()[1].index)):
            
                with col_wwr[metric]:
                    st.metric(f"WWR-{facade_calc()[1].index[metric]}",f"{int(facade_calc()[1].iloc[metric].values/facade_calc()[2].iloc[metric].values*100)}%")
        
        with cols[1]:
            st.dataframe(facade_calc()[1], use_container_width=True)
        with cols[2]:
            st.dataframe(facade_calc()[2], use_container_width=True)
        with cols[3]:
            st.dataframe(facade_calc()[3], use_container_width=True)
            st.dataframe(facade_calc()[4], use_container_width=True)

    st.markdown('---')

else:
    ""

