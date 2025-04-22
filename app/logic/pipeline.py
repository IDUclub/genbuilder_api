from loguru import logger
from app.logic.generation import builder
from app.logic.vectorization import vectorizer
from app.logic.postprocessing import postprocessing


class GenbuilderPipeline:
    async def run(
        self,
        image_size,
        input_zones,
        prompt_dict,
        num_steps,
        guidance_scale
    ):
        """
        Запускает полный конвейер: генерацию, векторизацию и постобработку.
        """
        logger.info(
            "Starting GenbuilderPipeline with image_size={}, num_steps={}, guidance_scale={}",
            image_size,
            num_steps,
            guidance_scale
        )
        generated_image = await builder.generate_buildings(
            image_size,
            input_zones,
            prompt_dict,
            num_steps,
            guidance_scale
        )
        logger.info(
            "Generated image for {} zones", len(input_zones)
        )

        buildings_gdf = await vectorizer.vectorize_buildings(
            input_zones,
            generated_image
        )
        logger.info(
            "Vectorized buildings: {} features", len(buildings_gdf)
        )

        processed_gdf = await postprocessing.process_buildings(buildings_gdf)
        logger.info(
            "Postprocessing completed: {} features", len(processed_gdf)
        )

        logger.info("GenbuilderPipeline completed successfully.")
        return processed_gdf


genbuilder_pipe = GenbuilderPipeline()
