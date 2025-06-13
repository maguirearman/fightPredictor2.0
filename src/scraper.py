"""
Enhanced MMA Data Scraper for UFC Fight Predictor 2.0
Combines UFC.com, Sherdog, and UFC Stats data sources
"""

import requests
from lxml import html
import pandas as pd
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import datetime as dt
import re
from pathlib import Path
import json

from config import SCRAPING_CONFIG, DATA_SOURCES, FILE_PATHS, VALIDATION_RULES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Fighter:
    """Fighter data structure"""
    name: str
    nickname: str = ""
    nationality: str = ""
    height: str = ""
    weight: str = ""
    reach: str = ""
    age: str = ""
    record: str = ""
    weight_class: str = ""
    wins_total: int = 0
    losses_total: int = 0
    wins_ko: int = 0
    wins_sub: int = 0
    wins_dec: int = 0
    losses_ko: int = 0
    losses_sub: int = 0
    losses_dec: int = 0

@dataclass
class Fight:
    """Fight data structure"""
    event_name: str
    event_date: str
    fighter1: str
    fighter2: str
    winner: str
    method: str
    round_finished: str
    time: str
    weight_class: str
    location: str = ""
    
@dataclass
class FightStats:
    """Detailed fight statistics"""
    fighter_name: str
    fight_id: str
    knockdowns: int = 0
    sig_strikes_landed: int = 0
    sig_strikes_attempted: int = 0
    total_strikes_landed: int = 0
    total_strikes_attempted: int = 0
    takedowns_landed: int = 0
    takedowns_attempted: int = 0
    submission_attempts: int = 0
    control_time_seconds: int = 0
    striking_accuracy: float = 0.0
    takedown_accuracy: float = 0.0

class EnhancedMMAScaper:
    """Enhanced MMA scraper combining multiple data sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': SCRAPING_CONFIG['user_agent']
        })
        self.fighters_cache = {}
        self.fights_cache = []
        self.stats_cache = []
        
    def _make_request(self, url: str, max_retries: int = None) -> Optional[html.HtmlElement]:
        """Make HTTP request with retry logic"""
        max_retries = max_retries or SCRAPING_CONFIG['max_retries']
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url, 
                    timeout=SCRAPING_CONFIG['request_timeout']
                )
                response.raise_for_status()
                return html.document_fromstring(response.content)
            except Exception as e:
                logger.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(SCRAPING_CONFIG['delay_between_requests'] * (attempt + 1))
                else:
                    logger.error(f"All requests failed for {url}")
                    return None
    
    def _search_google(self, query: str) -> List[str]:
        """Search Google for relevant links"""
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            xml = self._make_request(search_url)
            if xml is None:
                return []
            
            links = xml.xpath("//h3/parent::a/@href")
            return [link for link in links if link.startswith('http')]
        except Exception as e:
            logger.error(f"Google search failed for '{query}': {e}")
            return []
    
    def _get_sherdog_link(self, fighter_name: str) -> Optional[str]:
        """Find Sherdog profile link for a fighter"""
        query = f"{fighter_name} Sherdog"
        possible_urls = self._search_google(query)
        
        for url in possible_urls:
            if "sherdog.com/fighter/" in url and "/news/" not in url:
                return url
        
        logger.warning(f"Sherdog link not found for {fighter_name}")
        return None
    
    def _get_ufc_link(self, fighter_name: str) -> Optional[str]:
        """Find UFC.com profile link for a fighter"""
        query = f"{fighter_name} UFC.com"
        possible_urls = self._search_google(query)
        
        for url in possible_urls:
            if "ufc.com/athlete/" in url:
                return url
        
        logger.warning(f"UFC link not found for {fighter_name}")
        return None
    
    def scrape_sherdog_fighter(self, url: str) -> Optional[Fighter]:
        """Scrape fighter data from Sherdog"""
        try:
            xml = self._make_request(url)
            if xml is None:
                return None
            
            # Extract basic info
            name = self._safe_xpath(xml, "//span[@class='fn']/text()", "Unknown")
            nickname = self._safe_xpath(xml, "//span[@class='nickname']/em/text()", "")
            nationality = self._safe_xpath(xml, "//strong[@itemprop='nationality']/text()", "Unknown")
            height = self._safe_xpath(xml, "//b[@itemprop='height']/text()", "")
            weight = self._safe_xpath(xml, "//b[@itemprop='weight']/text()", "")
            
            # Extract win/loss details
            wins_detailed = xml.xpath("//div[@class='wins']/div[@class='meter']/div[1]/text()")
            losses_detailed = xml.xpath("//div[@class='loses']/div[@class='meter']/div[1]/text()")
            
            # Parse wins
            wins_total = self._safe_xpath(xml, "//div[@class='winloses win']/span[2]/text()", "0")
            wins_ko = int(wins_detailed[0]) if len(wins_detailed) > 0 else 0
            wins_sub = int(wins_detailed[1]) if len(wins_detailed) > 1 else 0
            wins_dec = int(wins_detailed[2]) if len(wins_detailed) > 2 else 0
            
            # Parse losses
            losses_total = self._safe_xpath(xml, "//div[@class='winloses lose']/span[2]/text()", "0")
            losses_ko = int(losses_detailed[0]) if len(losses_detailed) > 0 else 0
            losses_sub = int(losses_detailed[1]) if len(losses_detailed) > 1 else 0
            losses_dec = int(losses_detailed[2]) if len(losses_detailed) > 2 else 0
            
            fighter = Fighter(
                name=name,
                nickname=nickname,
                nationality=nationality,
                height=height,
                weight=weight,
                wins_total=int(wins_total) if wins_total.isdigit() else 0,
                losses_total=int(losses_total) if losses_total.isdigit() else 0,
                wins_ko=wins_ko,
                wins_sub=wins_sub,
                wins_dec=wins_dec,
                losses_ko=losses_ko,
                losses_sub=losses_sub,
                losses_dec=losses_dec,
                record=f"{wins_total}-{losses_total}"
            )
            
            return fighter
            
        except Exception as e:
            logger.error(f"Error scraping Sherdog fighter {url}: {e}")
            return None
    
    def scrape_ufc_stats(self, url: str) -> Dict:
        """Scrape UFC official statistics"""
        try:
            xml = self._make_request(url)
            if xml is None:
                return {}
            
            # Extract striking stats
            distance_stats = xml.xpath("//div[@class='c-stat-3bar__value']/text()")
            general_stats = xml.xpath("//div[@class='c-stat-compare__number']/text()")
            
            # Extract detailed stats
            detailed_stats = []
            for item in xml.xpath("//dd"):
                detailed_stats.append(item.text if item.text else "0")
            
            stats = {
                'strikes_landed': detailed_stats[0] if len(detailed_stats) > 0 else "0",
                'strikes_attempted': detailed_stats[1] if len(detailed_stats) > 1 else "0",
                'takedowns_landed': detailed_stats[2] if len(detailed_stats) > 2 else "0",
                'takedowns_attempted': detailed_stats[3] if len(detailed_stats) > 3 else "0",
                'striking_accuracy': general_stats[0].strip() if len(general_stats) > 0 else "0%",
                'strikes_per_minute': general_stats[1].strip() if len(general_stats) > 1 else "0",
                'takedown_accuracy': general_stats[2].strip() if len(general_stats) > 2 else "0%",
                'takedown_defense': general_stats[3].strip() if len(general_stats) > 3 else "0%",
                'standing_strikes': distance_stats[0].split()[0] if len(distance_stats) > 0 else "0",
                'clinch_strikes': distance_stats[1].split()[0] if len(distance_stats) > 1 else "0",
                'ground_strikes': distance_stats[2].split()[0] if len(distance_stats) > 2 else "0"
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error scraping UFC stats {url}: {e}")
            return {}
    
    def scrape_ufc_events(self, num_events: int = 20) -> List[Fight]:
        """Scrape recent UFC events"""
        try:
            # Get upcoming/recent events from UFC.com
            events_url = f"{DATA_SOURCES['ufc_official']}/events"
            xml = self._make_request(events_url)
            if xml is None:
                return []
            
            # Extract event links
            event_links = xml.xpath("//details[@id='events-list-completed']/div/div/div/div/div/section/ul/li/article/div[1]/div/a/@href")
            event_links = [f"{DATA_SOURCES['ufc_official']}{link}" for link in event_links[:num_events]]
            
            all_fights = []
            for event_url in event_links:
                fights = self._scrape_single_event(event_url)
                all_fights.extend(fights)
                time.sleep(SCRAPING_CONFIG['delay_between_requests'])
            
            return all_fights
            
        except Exception as e:
            logger.error(f"Error scraping UFC events: {e}")
            return []
    
    def _scrape_single_event(self, event_url: str) -> List[Fight]:
        """Scrape fights from a single UFC event"""
        try:
            xml = self._make_request(event_url)
            if xml is None:
                return []
            
            # Extract event details
            event_name = self._safe_xpath(xml, "//div[@class='c-hero__header']/div[1]/div/h1/text()", "Unknown Event")
            event_date = self._safe_xpath(xml, "//div[@class='c-hero__bottom-text']/div[1]/@data-timestamp", "")
            
            if event_date:
                event_date = dt.datetime.fromtimestamp(int(event_date)).strftime("%Y-%m-%d")
            
            # Extract fights
            fight_elements = xml.xpath("//div[@class='fight-card']/div/div/section/ul/li")
            fights = []
            
            for fight_elem in fight_elements:
                try:
                    # Extract fighter names
                    fighter1 = self._get_fighter_name(fight_elem, 'red')
                    fighter2 = self._get_fighter_name(fight_elem, 'blue')
                    
                    if not fighter1 or not fighter2:
                        continue
                    
                    # Extract fight details
                    weight_class = self._safe_xpath(fight_elem, "div/div/div/div[2]/div[2]/div[1]/div[2]/text()", "Unknown")
                    weight_class = weight_class.replace(" Bout", "").strip()
                    
                    # Extract result (for completed fights)
                    results = fight_elem.xpath("div/div/div/div[2]//div[@class='c-listing-fight__outcome-wrapper']/div/text()")
                    method = self._safe_xpath(fight_elem, "div//div[@class='c-listing-fight__result-text method']/text()", "Decision")
                    round_finished = self._safe_xpath(fight_elem, "div//div[@class='c-listing-fight__result-text round']/text()", "3")
                    fight_time = self._safe_xpath(fight_elem, "div//div[@class='c-listing-fight__result-text time']/text()", "5:00")
                    
                    # Determine winner
                    winner = "Unknown"
                    if len(results) >= 2:
                        if "W" in results[0]:
                            winner = fighter1
                        elif "W" in results[1]:
                            winner = fighter2
                    
                    fight = Fight(
                        event_name=event_name.strip(),
                        event_date=event_date,
                        fighter1=fighter1,
                        fighter2=fighter2,
                        winner=winner,
                        method=method.strip(),
                        round_finished=round_finished.strip(),
                        time=fight_time.strip(),
                        weight_class=weight_class
                    )
                    
                    fights.append(fight)
                    
                except Exception as e:
                    logger.warning(f"Error parsing individual fight: {e}")
                    continue
            
            logger.info(f"Scraped {len(fights)} fights from {event_name}")
            return fights
            
        except Exception as e:
            logger.error(f"Error scraping event {event_url}: {e}")
            return []
    
    def _get_fighter_name(self, fight_elem, corner: str) -> str:
        """Extract fighter name from fight element"""
        if corner == 'red':
            xpath = "div/div/div/div[2]/div[2]/div[5]/div[1]/a/span/text()"
        else:
            xpath = "div/div/div/div[2]/div[2]/div[5]/div[3]/a/span/text()"
        
        name_parts = fight_elem.xpath(xpath)
        name = " ".join(name_parts).strip()
        
        if not name:
            # Try alternative xpath
            xpath_alt = xpath.replace("/span", "")
            name_parts = fight_elem.xpath(xpath_alt)
            name = " ".join(name_parts).strip()
        
        return name
    
    def _safe_xpath(self, xml, xpath: str, default: str = "") -> str:
        """Safely extract text using xpath"""
        try:
            result = xml.xpath(xpath)
            return result[0].strip() if result else default
        except (IndexError, AttributeError):
            return default
    
    def get_fighter_complete_profile(self, fighter_name: str) -> Optional[Fighter]:
        """Get complete fighter profile from multiple sources"""
        if fighter_name in self.fighters_cache:
            return self.fighters_cache[fighter_name]
        
        try:
            # Start with Sherdog data
            sherdog_url = self._get_sherdog_link(fighter_name)
            fighter = None
            
            if sherdog_url:
                fighter = self.scrape_sherdog_fighter(sherdog_url)
            
            if not fighter:
                # Create basic fighter profile
                fighter = Fighter(name=fighter_name)
            
            # Enhance with UFC stats
            ufc_url = self._get_ufc_link(fighter_name)
            if ufc_url:
                ufc_stats = self.scrape_ufc_stats(ufc_url)
                # Add UFC stats to fighter profile (you can extend Fighter class)
            
            self.fighters_cache[fighter_name] = fighter
            time.sleep(SCRAPING_CONFIG['delay_between_requests'])
            
            return fighter
            
        except Exception as e:
            logger.error(f"Error getting complete profile for {fighter_name}: {e}")
            return None
    
    def scrape_all_data(self, num_events: int = 20) -> Dict[str, pd.DataFrame]:
        """Main method to scrape all MMA data"""
        logger.info(f"Starting comprehensive data scraping for {num_events} events...")
        
        # Step 1: Scrape recent fights
        logger.info("Scraping recent UFC events...")
        fights = self.scrape_ufc_events(num_events)
        
        # Step 2: Get unique fighters from fights
        unique_fighters = set()
        for fight in fights:
            unique_fighters.add(fight.fighter1)
            unique_fighters.add(fight.fighter2)
        
        logger.info(f"Found {len(unique_fighters)} unique fighters")
        
        # Step 3: Scrape detailed fighter profiles
        fighters_data = []
        for i, fighter_name in enumerate(unique_fighters):
            if i >= SCRAPING_CONFIG['max_fighters_per_event']:
                break
                
            logger.info(f"Scraping fighter {i+1}/{min(len(unique_fighters), SCRAPING_CONFIG['max_fighters_per_event'])}: {fighter_name}")
            fighter = self.get_fighter_complete_profile(fighter_name)
            if fighter:
                fighters_data.append(fighter)
        
        # Step 4: Convert to DataFrames
        fights_df = pd.DataFrame([asdict(fight) for fight in fights])
        fighters_df = pd.DataFrame([asdict(fighter) for fighter in fighters_data])
        
        # Step 5: Generate mock fight stats (in real implementation, scrape from detailed fight pages)
        stats_data = self._generate_fight_stats(fights, fighters_data)
        stats_df = pd.DataFrame([asdict(stat) for stat in stats_data])
        
        logger.info(f"Scraping complete! Collected {len(fights)} fights, {len(fighters_data)} fighters, {len(stats_data)} stat records")
        
        return {
            'fights': fights_df,
            'fighters': fighters_df,
            'stats': stats_df
        }
    
    def _generate_fight_stats(self, fights: List[Fight], fighters: List[Fighter]) -> List[FightStats]:
        """Generate fight statistics (mock data for now)"""
        stats = []
        
        for i, fight in enumerate(fights):
            # Create stats for both fighters
            for fighter_name in [fight.fighter1, fight.fighter2]:
                # Generate realistic stats based on weight class and fighter profile
                weight_multiplier = self._get_weight_multiplier(fight.weight_class)
                
                stat = FightStats(
                    fighter_name=fighter_name,
                    fight_id=f"fight_{i}_{fighter_name.replace(' ', '_')}",
                    knockdowns=np.random.poisson(0.3 * weight_multiplier),
                    sig_strikes_landed=np.random.poisson(45 * weight_multiplier),
                    sig_strikes_attempted=np.random.poisson(85 * weight_multiplier),
                    total_strikes_landed=np.random.poisson(60 * weight_multiplier),
                    total_strikes_attempted=np.random.poisson(110 * weight_multiplier),
                    takedowns_landed=np.random.poisson(1.2),
                    takedowns_attempted=np.random.poisson(3.5),
                    submission_attempts=np.random.poisson(0.8),
                    control_time_seconds=np.random.randint(0, 600)
                )
                
                # Calculate derived stats
                if stat.sig_strikes_attempted > 0:
                    stat.striking_accuracy = stat.sig_strikes_landed / stat.sig_strikes_attempted
                if stat.takedowns_attempted > 0:
                    stat.takedown_accuracy = stat.takedowns_landed / stat.takedowns_attempted
                
                stats.append(stat)
        
        return stats
    
    def _get_weight_multiplier(self, weight_class: str) -> float:
        """Get multiplier based on weight class for realistic stats"""
        multipliers = {
            'Heavyweight': 1.2,
            'Light Heavyweight': 1.1,
            'Middleweight': 1.0,
            'Welterweight': 0.95,
            'Lightweight': 0.9,
            'Featherweight': 0.85,
            'Bantamweight': 0.8,
            'Flyweight': 0.75
        }
        return multipliers.get(weight_class, 1.0)
    
    def save_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Save scraped data to CSV files"""
        try:
            data['fights'].to_csv(FILE_PATHS['raw_fights'], index=False)
            data['fighters'].to_csv(FILE_PATHS['raw_fighters'], index=False)
            data['stats'].to_csv(FILE_PATHS['raw_stats'], index=False)
            
            logger.info("Data saved successfully to CSV files")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def load_cached_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load previously scraped data"""
        try:
            fights_df = pd.read_csv(FILE_PATHS['raw_fights'])
            fighters_df = pd.read_csv(FILE_PATHS['raw_fighters'])
            stats_df = pd.read_csv(FILE_PATHS['raw_stats'])
            
            return {
                'fights': fights_df,
                'fighters': fighters_df,
                'stats': stats_df
            }
            
        except FileNotFoundError:
            logger.info("No cached data found")
            return None
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            return None

# Convenience functions for easy usage
def scrape_fresh_data(num_events: int = 20, save: bool = True) -> Dict[str, pd.DataFrame]:
    """Scrape fresh data and optionally save it"""
    scraper = EnhancedMMAScaper()
    data = scraper.scrape_all_data(num_events)
    
    if save:
        scraper.save_data(data)
    
    return data

def load_or_scrape_data(num_events: int = 20) -> Dict[str, pd.DataFrame]:
    """Load cached data or scrape fresh data if none exists"""
    scraper = EnhancedMMAScaper()
    data = scraper.load_cached_data()
    
    if data is None:
        logger.info("No cached data found, scraping fresh data...")
        data = scraper.scrape_all_data(num_events)
        scraper.save_data(data)
    
    return data

if __name__ == "__main__":
    # Test the scraper
    print("Testing MMA Data Scraper...")
    
    # Scrape some data
    data = scrape_fresh_data(num_events=5)
    
    print(f"\nResults:")
    print(f"Fights: {len(data['fights'])} records")
    print(f"Fighters: {len(data['fighters'])} records")
    print(f"Stats: {len(data['stats'])} records")
    
    if len(data['fights']) > 0:
        print(f"\nSample fight: {data['fights'].iloc[0]['event_name']}")
        print(f"Sample fighter: {data['fighters'].iloc[0]['name']}")
    
    print("\nScraper test completed!")